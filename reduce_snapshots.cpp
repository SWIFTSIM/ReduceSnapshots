/**
 * @file reduce_snapshots.cpp
 *
 * @brief Program that takes a SOAP halo catalogue and a SWIFT snapshot and
 * creates a new (SWIFT compatible) snapshot that only contains the particles
 * that are part of spherical overdensities (SOs) of the flagged halos.
 *
 * Compilation requires MPI (any version) and HDF5 (>1.10.0).
 * Example compilation commands:
 * - on Ubuntu 18.04):
 *  mpic++ -Wall -Werror -fopenmp -O3 -I/usr/include/hdf5/serial \
 *   -o reduce_snapshots reduce_snapshots.cpp \
 *   -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
 * - on cosma:
 *  module load intel_comp/2020-update2 \
 *    intel_mpi/2020-update2 \
 *    parallel_hdf5/1.10.6 gsl/2.5
 *  mpiicpc -O3 -qopenmp -qoverride-limits \
 *   -I/cosma/local/parallel-hdf5//intel_2020-update2_intel_mpi_2020-update2/1.10.6/include
 * \
 *   -o reduce_snapshots reduce_snapshots.cpp \
 *   -L/cosma/local/parallel-hdf5//intel_2020-update2_intel_mpi_2020-update2/1.10.6/lib
 * \
 *   -Wl,-rpath=/cosma/local/parallel-hdf5//intel_2020-update2_intel_mpi_2020-update2/1.10.6/lib
 * \ -lhdf5
 *
 * Example run command:
 *  mpirun -np 16 ./reduce_snapshots halos_0008 flamingo_0008/flamingo_0008 \
 *    test_0008 SO/50_crit/SORadius 512
 * Where all arguments are positional and:
 *  - halos_0008 is the prefix for the halo catalogue files
 *    (we require halos_0008.siminfo, halos_0008.properties[.*],
 *     halos_0008.catalog_SOlist[.*])
 *  - flamingo_0008/flamingo_0008 is the prefix for a (distributed) SWIFT
 *    snapshot file (we require at least flamingo_0008/flamingo_0008.hdf5)
 *  - test_0008 is the prefix for the output snapshot files
 *    (we will create the same number of files as the input files, using the
 *     same indices if it is a distributed snapshot file)
 *  - SO_R_100_rhocrit is the radius used to determine if a particle belongs to
 *    an SO. 
 *  - 512 is the number of SWIFT top-level cells that are processed in one go.
 *    This should be a proper divisor of the total number of top-level cells
 *    (an error is thrown if this is not the case). A larger number leads to
 *    more memory usage and more complex table lookup operations. A lower
 *    number leads to more frequent disk reads and less continguous OpenMP
 *    parallel regions. If you can afford it memory-wise, a larger number will
 *    lead to better performance.
 *
 * Additional optional arguments:
 *  - HDF5 BLOCK SIZE: Maximum size (in bytes) that can be read while copying
 *    datasets from the original snapshot to the new file. A larger number
 *    uses more memory. The default is to read entire datasets. Note that
 *    applying a low limit here can lead to significant changes in dataset
 *    values if lossy compression is used, since the compression is applied
 *    on the individual chunks.
 *
 * @author Bert Vandenbroucke (vandenbroucke@strw.leidenuniv.nl)
 * @author Rob McGibbon (mcgibbon@strw.leidenuniv.nl)
 */

// local includes:
// - custom error messages and assertions
#include "Error.hpp"
// - custom HDF5 library wrappers
#include "HDF5.hpp"

// standard includes
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <tuple>
#include <vector>

// we expose all HDF5 functions
using namespace HDF5;
namespace fs = std::experimental::filesystem;

// activate this to set Lustre striping to 1 on new output files
#define SET_STRIPING

// Available log levels:
//  - output log messages from innermost loops
#define LOGLEVEL_FILELOOPS 2
//  - output log messages from first level loops
#define LOGLEVEL_CHUNKLOOPS 1
//  - only output general progress messages
#define LOGLEVEL_GENERAL 0

// set the log level for output
// a higher value means more output
#define LOG_LEVEL 1

// uncomment this to output detailed memory logs
//#define OUTPUT_MEMORY_LOG

/**
 * @brief Simple execution timer used to display program progress.
 */
class Timer {
private:
  /*! @brief Start time of the timer. */
  timeval _start;

public:
  /**
   * @brief Start the timer.
   */
  inline void start() { gettimeofday(&_start, nullptr); }

  /**
   * @brief Constructor.
   *
   * Starts the timer.
   */
  inline Timer() { start(); }

  /**
   * @brief Get the current running time of the timer.
   *
   * @return Time since the timer was (last) started (in s).
   */
  inline double current_time() {
    timeval stop;
    gettimeofday(&stop, nullptr);
    timeval result;
    timersub(&stop, &_start, &result);
    return result.tv_sec + 1.e-6 * result.tv_usec;
  }
};

// global timer (made global so that we can display progress messages from
// anywhere within the program)
Timer global_timer;
// global MPI control variables (global so that they can be displayed from
// anywhere within the program)
int MPI_rank, MPI_size;

/**
 * @brief Write a progress message to the stderr.
 *
 * The message will include the current program run time (in seconds), the rank
 * of the MPI process that writes and the location of the line that called the
 * macro.
 *
 * @param s Message string (can contain format characters, like printf()).
 * @param ... Additional arguments passed on to printf() (should match the
 * format specifiers).
 */
#define timelog(level, s, ...)                                                 \
  {                                                                            \
    if (level <= LOG_LEVEL) {                                                  \
      FILE *handle = (LOG_LEVEL == 0) ? stdout : stderr;                       \
      fprintf(handle, "[%9.4f] [%3i]", global_timer.current_time(), MPI_rank); \
      fprintf(handle, "%s:%s():%i: ", __FILE__, __FUNCTION__, __LINE__);       \
      fprintf(handle, s "\n", ##__VA_ARGS__);                                  \
    }                                                                          \
  }

/**
 * @brief Find a file composed of a prefix, a name and a suffix and containing
 * an optional index number.
 *
 * With prefix = "A", name = "B", suffix = ".C" and index = 1, the function
 * will look for the following files:
 *  - regular file: AB.C
 *  - indexed file: AB.1.C
 * (note that the index is always preceded by a '.').
 * If none of these files are found, an error is thrown. In other cases, the
 * behaviour depends on the values of the arguments only_one and use_index:
 *  - only_one = true, use_index = true (default): if both files exist, an
 *    error is thrown. If only one of them exists, then the name of the
 *    existing file is returned.
 *  - only_one = false, use_index = true: the name of the distributed file is
 *    returned if it exists, otherwise the name of the regular file is
 *    returned.
 *  - use_index = false: the name of the regular file is returned if it exists,
 *    an error is thrown if it does not. The distributed file is ignored.
 *
 * Note that the prefix and name are always combined in the same way. The only
 * reason they are provided as separate arguments is to make it easier to
 * compose filenames where the prefix and name are constructed in different
 * ways.
 *
 * @param prefix File prefix.
 * @param name File name.
 * @param suffix File suffix (default: "").
 * @param index File index (default: 0).
 * @param only_one Require only one of the possible file matches to be present
 * (default: yes - see above).
 * @param use_index False if the index can be ignored; we only care about one
 * of the possible file matches (default: true - see above).
 * @return File name. The file will have been successfully opened, meaning it
 * should exist.
 */
inline std::string find_file(const std::string prefix, const std::string name,
                             const std::string suffix = "",
                             const int_fast32_t index = 0,
                             const bool only_one = true,
                             const bool use_index = true) {
  std::stringstream filename;
  filename << prefix << name << suffix;
  std::ifstream file(filename.str().c_str());
  std::string non_distributed;
  if (file.good()) {
    non_distributed = filename.str();
    file.close();
  }
  if (!use_index) {
    // we explicitly ask for the non-distributed file (might be a virtual file)
    if (non_distributed.size() == 0) {
      my_error("Unable to find \"%s%s%s\"!", prefix.c_str(), name.c_str(),
               suffix.c_str());
    }
    return non_distributed;
  }
  std::string distributed;
  filename.str("");
  filename << prefix << name << "." << index << suffix;
  file.open(filename.str().c_str());
  if (file.good()) {
    distributed = filename.str();
    file.close();
  }
  if (distributed.size() == 0 && non_distributed.size() == 0) {
    my_error("Unable to find \"%s%s%s\" or \"%s%s.%" PRIiFAST32 "%s\"!",
             prefix.c_str(), name.c_str(), suffix.c_str(), prefix.c_str(),
             name.c_str(), index, suffix.c_str());
  } else {
    if (only_one && distributed.size() > 0 && non_distributed.size() > 0) {
      my_error("Found both \"%s%s%s\" and \"%s%s.%" PRIiFAST32 "%s\"!",
               prefix.c_str(), name.c_str(), suffix.c_str(), prefix.c_str(),
               name.c_str(), index, suffix.c_str());
    }
  }
  // if both files exist, we favour the distributed one (unless only_one is set,
  // in which case we already threw an error)
  return (distributed.size() > 0) ? distributed : non_distributed;
}

/**
 * @brief Expand a dataset path into a list of groups and the dataset name.
 *
 * E.g. "path/to/dataset" --> (["path, "to"], "dataset")
 *      "dataset" --> ([], "dataset")
 *
 * @param path Full path to the dataset.
 * @return std::pair containing a std::vector with all groups in the path,
 * and the name of the dataset.
 */
inline std::pair<std::vector<std::string>, std::string>
decompose_dataset_path(const std::string path) {
  size_t prevpos = 0;
  size_t pos = path.find("/");
  std::vector<std::string> full_path;
  while (pos != std::string::npos) {
    const std::string path_part = path.substr(prevpos, pos - prevpos);
    full_path.push_back(path_part);
    prevpos = pos + 1;
    pos = path.find("/", prevpos);
  }
  const std::string dset_name = path.substr(prevpos, pos);
  return std::make_pair(full_path, dset_name);
}

/**
 * @brief Compose a file name using the given prefix and suffix and an optional
 * additional index.
 *
 * If prefix = "A", suffix = ".B" and index = 1, this function returns
 *  - "A.B" if use_index is false (default)
 *  - "A.1.B" if use_index is true
 * (note that the index is always preceded by a '.').
 *
 * If use_index is false, an index value other than 0 will lead to an error.
 *
 * @param prefix File name prefix.
 * @param suffix File name suffix.
 * @param index Optional index (default: 0).
 * @param use_index Whether or not to use the index (default: no).
 * @return Full composed file name.
 */
inline std::string compose_filename(const std::string prefix,
                                    const std::string suffix,
                                    const int_fast32_t index = 0,
                                    const bool use_index = false) {
  std::stringstream filename;
  // sanity check: if we set the index to something other than 0, we probably
  // intend to use it
  if (!use_index && index > 0) {
    my_error("Cannot create a file with index %" PRIiFAST32
             " when use_index is disabled!",
             index);
  }
  if (use_index) {
    filename << prefix << "." << index << suffix;
  } else {
    filename << prefix << suffix;
  }
  return filename.str();
}


/**
 * @brief Convert a size in bytes to something a human understands.
 *
 * This function is limited to a size of 16 EB, since that is the largest value
 * that fits in a size_t.
 *
 * @param size Size in bytes.
 * @return String representation of the size with two digit precision and the
 * closest smaller human-readable unit.
 */
inline std::string human_readable_bytes(size_t size) {
  const std::string units[7] = {"bytes", "KB", "MB", "GB", "TB", "PB", "EB"};
  double floatsize = size;
  int_fast8_t iu = 0;
  // we don't go beyond EB, since 16 EB is the maximum size_t
  while (iu < 7 && size > 0) {
    floatsize /= 1024.;
    size >>= 10;
    ++iu;
  }
  // correct for the overshoot
  --iu;
  floatsize *= 1024.;
  char sizestr[100];
  sprintf(sizestr, "%.2f %s", floatsize, units[iu].c_str());
  return std::string(sizestr);
}

/**
 * @brief Utility class used to keep track of memory usage.
 *
 * The memory log file is a hierarchical structure consisting of at least a root
 * level group. Each group in this hierarchy can contain multiple other groups
 * and individual memory allocation entries. These are all written to a text
 * file. Groups can have two types: a normal group simply accumulates memory
 * allocations for its elements. A "loop scope" group instead assumes only one
 * of its elements is contained in memory at the same time and reports the
 * maximum along its elements as memory usage.
 *
 * Internally, groups are represented by a MemoryLogBlock. No MemoryLogBlocks
 * are actually stored within the MemoryLog. Upon creation, the MemoryLog will
 * store an entry marking the MemoryLogBlock, while the MemoryLogBlock stores
 * the information required to link its elements in the log file. All access
 * to the MemoryLog should happen through the MemoryLogBlocks.
 *
 * As an example, the following code:
 *  int main(){
 *    MemoryLog mlog(100);
 *    auto mroot = mlog.get_root();
 *
 *    std::vector<double> a(400);
 *    mroot.add_entry("a", a);
 *    mroot.add_entry("random size", 100);
 *
 *    auto mloop = mroot.add_loop_scope("loop");
 *    for(int i = 0; i < 4; ++i){
 *      std::vector<double> b(100);
 *      auto mloopit = mloop.add_block("iteration");
 *      mloopit.add_entry("b", b);
 *    };
 *
 *    mlog.dump("test.txt");
 *
 *    return 0;
 *  }
 * will produce the following memory log (test.txt):
 *  # type  name  size (bytes)  parent
 *  g  Root  0  0
 *  e  a  3200 0
 *  e  random size  100  0
 *  l  loop  0  0
 *  g  iteration  0  3
 *  e  b  400  4
 *  g  iteration  0  3
 *  e  b  400  6
 *  g  iteration  0  3
 *  e  b  400  8
 *  g  iteration  0  3
 *  e  b  400  10
 * Which corresponds to the following hierarchy:
 *  - Root:
 *    - a: 3200
 *    - random size: 100
 *    - loop:
 *      - iteration:
 *        - b: 400
 *      - iteration:
 *        - b: 400
 *      - iteration:
 *        - b: 400
 *      - iteration:
 *        - b: 400
 * When analysed with the appropriate Python script, the total memory size will
 * be found to be 3200 + 100 + 400.
 */
class MemoryLog {
public:
  // forward declaration of the member class
  class MemoryLogBlock;

private:
  /*! @brief Lines in the log. Each line consists of a type char, a name string
   *  a size entry and a parent reference. The parent reference should refer
   *  to another line in the log. */
  std::vector<std::tuple<char, std::string, size_t, size_t>> _log;
  /*! @brief Next available entry in the log. The log is preallocated to
   *  minimise reallocations, so that we have to store the current index
   *  separately. */
  size_t _next_entry;

  /**
   * @brief Add an entry to the log.
   *
   * @param type Type of the entry (currently accepted: 'e', 'g', 'l').
   * @param name Name used to identify the entry. Spaces in the name will be
   * replaced by '_'.
   * @param size Size (in bytes) that should be logged.
   * @param parent Line on which the entry for the parent group was made.
   */
  inline size_t add_entry(const char type, const std::string name,
                          const size_t size, const size_t parent) {
    // check if we need to grow the log
    if (_next_entry == _log.size()) {
      _log.resize(2 * _log.size());
    }
    std::string name_nospace(name);
    // replace whitespaces in the name with '_' (to help numpy.loadtxt)
    std::transform(name_nospace.begin(), name_nospace.end(),
                   name_nospace.begin(),
                   [](char c) { return (c == ' ') ? '_' : c; });
    // create the entry
    std::get<0>(_log[_next_entry]) = type;
    std::get<1>(_log[_next_entry]) = name_nospace;
    std::get<2>(_log[_next_entry]) = size;
    std::get<3>(_log[_next_entry]) = parent;
    // update the current index
    ++_next_entry;
    // return the line on which this entry was made (used for group references)
    return _next_entry - 1;
  }

public:
  /**
   * @brief Constructor.
   *
   * @param initial_size Initial size of the log. Should be sensible to avoid
   * having to grow the log all the time.
   */
  MemoryLog(const size_t initial_size)
      : _log(initial_size > 0 ? initial_size : 1), _next_entry(0) {
    add_entry('g', "Root", 0, 0);
  }

  /**
   * @brief Dump the log to a file with the given name.
   *
   * @param filename Output file name.
   */
  inline void dump(const std::string filename) {
    std::ofstream ofile(filename.c_str());
    ofile << "# type\tname\tsize (bytes)\tparent\n";
    for (size_t i = 0; i < _next_entry; ++i) {
      char type;
      std::string name;
      size_t size, parent;
      std::tie(type, name, size, parent) = _log[i];
      ofile << type << "\t" << name << "\t" << size << "\t" << parent << "\n";
    }
  }

  /**
   * @brief Block (group) in the memory log.
   *
   * A block acts as an external gateway into the log. Upon creation, it adds
   * an appropriate entry in the log. Whenever an entry is made in the block,
   * the block appropriately links that entry to its own entry by setting the
   * correct parent index. New levels in the hierarchy can be created by
   * adding child blocks to the block.
   */
  class MemoryLogBlock {
  private:
    /*! @brief Reference to the MemoryLog. */
    MemoryLog &_log;
    /*! @brief Line on which the block entry was made in the log. This number
     *  is used as parent index for all child entries. */
    const size_t _line;

  public:
    /**
     * @brief Constructor.
     *
     * @param log Reference to the MemoryLog.
     * @param line Line on which the block entry was made in the log.
     */
    inline MemoryLogBlock(MemoryLog &log, const size_t line)
        : _log(log), _line(line) {}

    /**
     * @brief Add a child block to this block.
     *
     * @param name Name of the block in the log file.
     * @return New MemoryLogBlock representing the new block.
     */
    inline MemoryLogBlock add_block(const std::string name) {
      const size_t line = _log.add_entry('g', name, 0, _line);
      return MemoryLogBlock(_log, line);
    }

    /**
     * @brief Add a child "loop scope" block to this block.
     *
     * @param name Name of the block in the log file.
     * @return New MemoryLogBlock representing the new block.
     */
    inline MemoryLogBlock add_loop_scope(const std::string name) {
      const size_t line = _log.add_entry('l', name, 0, _line);
      return MemoryLogBlock(_log, line);
    }

    /**
     * @brief Log the memory used by a vector.
     *
     * @tparam _type_ Variable type stored in the vector.
     * @param name Name of the entry in the log file.
     * @param v Reference to the vector.
     */
    template <typename _type_>
    inline void add_entry(const std::string name,
                          const std::vector<_type_> &v) {
      const size_t size = sizeof(_type_) * v.size();
      _log.add_entry('e', name, size, _line);
    }

    /**
     * @brief Log the memory used by a multimap.
     *
     * @tparam _type_A_ Type of the keys in the multimap.
     * @tparam _type_B_ Type of the values in the multimap.
     * @param name Name of the entry in the log file.
     * @param mm Reference to the multimap.
     */
    template <typename _type_A_, typename _type_B_>
    inline void add_entry(const std::string name,
                          const std::multimap<_type_A_, _type_B_> &mm) {
      const size_t size = (sizeof(_type_A_) + sizeof(_type_B_)) * mm.size();
      _log.add_entry('e', name, size, _line);
    }

    /**
     * @brief Log a custom memory size in the log.
     *
     * @param name Name of the entry in the log file.
     * @param size Size (in bytes) that needs to be logged.
     */
    inline void add_entry(const std::string name, const size_t size) {
      _log.add_entry('e', name, size, _line);
    }
  };

  /**
   * @brief Get the MemoryLogBlock that represents the root of the log.
   *
   * Since MemoryLogBlocks are not actually stored in the MemoryLog, every
   * call to this function will return a different object. All objects will
   * however behave exactly the same as far as the log file hierarchy is
   * concerned.
   *
   * @return Root of the log.
   */
  inline MemoryLogBlock get_root() { return MemoryLogBlock(*this, 0); }
};

/**
 * @brief Internal representation of the SOs contained in a SOAP halo catalogue.
 *
 * Upon creation, the SOTable object will read the relevant SOAP catalogue files
 * and store all the information that is required for later access. To minimise
 * memory usage, only SOs that are filtered out are kept in memory.
 *
 */
class SOTable {
private:
  /*! @brief SO coordinates: x. */
  std::vector<double> _XSO;
  /*! @brief SO coordinates: y. */
  std::vector<double> _YSO;
  /*! @brief SO coordinates: z. */
  std::vector<double> _ZSO;
  /*! @brief SO radii. */
  std::vector<double> _RSO;
  /*! @brief IDs of the halos. */
  std::vector<uint64_t> _haloIDs;

  /*! @brief Offsets of the distributed files in the total SO property
   *  arrays. */
  std::vector<size_t> _file_boundaries;

  /// Catalogue stats
  /*! @brief Number of halos which are kept. */
  size_t _Nkeep;
  /*! @brief Number of halos in the catalogue (_Nkeep <= _Nhalo). */
  size_t _Nhalo;

  /*! @brief Prefix of the catalogue file names. */
  const std::string _prefix;

public:
  /**
   * @brief Constructor.
   *
   * Does the full parsing of the SOAP catalogue and can therefore require quite
   * some time and memory.
   *
   * @param prefix Prefix of the catalogue file names. We require the following
   * files to be present: prefix.properties[.*], prefix.siminfo,
   * prefix.catalog_SOlist[.*].
   * @param radius_selection_name Name of the array in the .properties file that
   * will be used for spatial particle filtering. Ideally, this name would
   * match SO_R_XXX_rhocrit, where XXX is the value of the VR configuration
   * variable Overdensity_output_maximum_radius_in_critical_density. But again,
   * any possible valid array would work (including those that do not represent
   * any radius at all, so be careful - again!).
   * @param memory MemoryLogBlock used to log memory usage.
   */
  SOTable(const std::string prefix, const std::string radius_selection_name,
          MemoryLog::MemoryLogBlock &memory)
      : _prefix(prefix) {

    timelog(LOGLEVEL_GENERAL, "Reading SOAP catalog...");

    const std::string first_catalog_file = find_file(prefix, "");
    // find out number of files
    HDF5FileOrGroup file = OpenFile(first_catalog_file, HDF5FileModeRead);
    const int32_t num_of_files = 1;
    HDF5FileOrGroup halogroup = OpenGroup(file, "InputHalos");
    const uint64_t refTotNhalo = GetDatasetSize(halogroup, "HaloCatalogueIndex");
    CloseGroup(halogroup);
    HDF5FileOrGroup cosmogroup = OpenGroup(file, "Cosmology");
    double raw_scale_factor[1];
    ReadArrayAttribute(cosmogroup, "Scale-factor", raw_scale_factor);
    const double scale_factor = raw_scale_factor[0];
    CloseGroup(cosmogroup);
    CloseFile(file);

    timelog(LOGLEVEL_GENERAL,
            "Found %zu halos in %i file(s). Scale factor is %g.", refTotNhalo,
            num_of_files, scale_factor);

    // read all the SO files and collect general catalogue statistics
    _Nhalo = 0;
    _Nkeep = 0;
    // file offsets used for various purposes
    // the bookkeeping in the loop is quite tedious!
    size_t keep_file_offset = 0;
    // register a loop scope in the memory log
    auto memory_ifile_loop = memory.add_loop_scope("File loop");
    for (int32_t ifile = 0; ifile < num_of_files; ++ifile) {
      // find the appropriate part of the catalogue
      const std::string catalog_file = find_file(prefix, "", "", ifile);
      HDF5FileOrGroup file = OpenFile(catalog_file, HDF5FileModeRead);

      // create a memory group for this iteration in the log
      std::stringstream ifilename;
      ifilename << "File" << ifile;
      auto memory_ifile = memory_ifile_loop.add_block(ifilename.str());

      // find out how many halos we are dealing with in this file
      const uint64_t num_of_groups = refTotNhalo;
      _Nhalo += num_of_groups;

      // now read all the relevant halo properties
      // we first read the entire file and then copy the appropriate bits
      // into the class arrays
      std::vector<int32_t> is_central(num_of_groups);
      std::vector<double> XSO(num_of_groups);
      std::vector<double> YSO(num_of_groups);
      std::vector<double> ZSO(num_of_groups);
      std::vector<double> CofP(3 * num_of_groups);
      std::vector<double> RSO(num_of_groups);
      std::vector<uint64_t> haloIDs(num_of_groups);
      std::vector<int> keep(num_of_groups, false);

      double unit_length_in_cgs;
      {
        HDF5FileOrGroup units = OpenGroup(file, "Units");
        double temp[1];
        ReadArrayAttribute(units, "Unit length in cgs (U_L)", temp);
        unit_length_in_cgs = temp[0];
        CloseGroup(units);
      }

      halogroup = OpenGroup(file, "InputHalos");
      ReadEntireDataset(halogroup, "IsCentral", is_central);
      ReadEntireDataset(halogroup, "HaloCatalogueIndex", haloIDs);
      ReadEntireDataset(halogroup, "HaloCentre", CofP);
      {
        // A conversion factor is needed because SOAP can be set to output in physical units
        double cgs_factor[1];
        HDF5FileOrGroup dset = OpenDataset(halogroup, "cofp");
        ReadArrayAttribute(
            dset,
            "Conversion factor to CGS (including cosmological corrections)",
            cgs_factor);
        CloseDataset(dset);
        const double conversion_factor = cgs_factor[0] / unit_length_in_cgs;
        for (size_t ihalo = 0; ihalo < num_of_groups; ++ihalo) {
          CofP[3 * ihalo] *= conversion_factor;
          CofP[3 * ihalo + 1] *= conversion_factor;
          CofP[3 * ihalo + 2] *= conversion_factor;
        }
      }
      CloseGroup(halogroup);

      for (size_t ihalo = 0; ihalo < num_of_groups; ++ihalo) {
        XSO[ihalo] = CofP[3 * ihalo];
        YSO[ihalo] = CofP[3 * ihalo + 1];
        ZSO[ihalo] = CofP[3 * ihalo + 2];
      }
      CofP.clear();

      HDF5FileOrGroup SOAPgroup = OpenGroup(file, "SOAP");
      ReadEntireDataset(SOAPgroup, "IncludedInReducedSnapshot", keep);

      {
        const auto radius_path = decompose_dataset_path(radius_selection_name);
        const auto group_names = radius_path.first;
        const auto radius_name = radius_path.second;
        HDF5FileOrGroup parent_group = file;
        std::vector<HDF5FileOrGroup> groups(group_names.size());
        for (uint_fast32_t igroup = 0; igroup < group_names.size(); ++igroup) {
          groups[igroup] = OpenGroup(parent_group, group_names[igroup]);
          parent_group = groups[igroup];
        }
        ReadEntireDataset(parent_group, radius_name, RSO);
        // A conversion factor is needed because SOAP can be set to output in physical units
        double cgs_factor[1];
        HDF5FileOrGroup dset = OpenDataset(parent_group, radius_name);
        ReadArrayAttribute(
            dset,
            "Conversion factor to CGS (including cosmological corrections)",
            cgs_factor);
        CloseDataset(dset);
        for (uint_fast32_t igroup = 0; igroup < group_names.size(); ++igroup) {
          CloseGroup(groups[igroup]);
        }
        const double conversion_factor = cgs_factor[0] / unit_length_in_cgs;
        for (size_t ihalo = 0; ihalo < num_of_groups; ++ihalo) {
          RSO[ihalo] *= conversion_factor;
        }
      }

      // we are done with the file, close it
      CloseFile(file);

      // report on memory usage
      memory_ifile.add_entry("is_central", is_central);
      memory_ifile.add_entry("XSO", XSO);
      memory_ifile.add_entry("YSO", YSO);
      memory_ifile.add_entry("ZSO", ZSO);
      memory_ifile.add_entry("RSO", RSO);
      memory_ifile.add_entry("haloIDs", haloIDs);
      memory_ifile.add_entry("keep", keep);

      // Count how many halos we are keeping
      uint64_t this_Nkeep = 0;
      for (size_t ih = 0; ih < num_of_groups; ++ih) {
        if (keep[ih] == 1) {
          ++this_Nkeep;
        }
      }
      // update the catalogue stats
      _Nkeep += this_Nkeep;

      // now that we know the size of this file, we can reallocate the class
      // arrays
      _XSO.resize(keep_file_offset + this_Nkeep, 0.);
      _YSO.resize(keep_file_offset + this_Nkeep, 0.);
      _ZSO.resize(keep_file_offset + this_Nkeep, 0.);
      _RSO.resize(keep_file_offset + this_Nkeep, 0.);
      _haloIDs.resize(keep_file_offset + this_Nkeep, 0);
      // copy the relevant SO properties
      size_t ikeep = 0;
      for (size_t ih = 0; ih < num_of_groups; ++ih) {
        if (keep[ih] == 1) {
          my_assert(RSO[ih] > 0., "Wrong RSO (%g)!", RSO[ih]);
          // convert distances from physical to co-moving
          // we need to do this because SWIFT outputs co-moving quantities
          _XSO[keep_file_offset + ikeep] = XSO[ih] / scale_factor;
          _YSO[keep_file_offset + ikeep] = YSO[ih] / scale_factor;
          _ZSO[keep_file_offset + ikeep] = ZSO[ih] / scale_factor;
          _RSO[keep_file_offset + ikeep] = RSO[ih] / scale_factor;
          _haloIDs[keep_file_offset + ikeep] = haloIDs[ih];
          ++ikeep;
        }
      }
    }

    timelog(LOGLEVEL_GENERAL, "Done reading SOAP catalog.");
    timelog(LOGLEVEL_GENERAL, "Stats: totNhalo: %zu, totNkeep: %zu", _Nhalo, _Nkeep);

    my_assert(_Nkeep == _XSO.size(), "Size mismatch!");
    my_assert(_Nkeep == _YSO.size(), "Size mismatch!");
    my_assert(_Nkeep == _ZSO.size(), "Size mismatch!");
    my_assert(_Nkeep == _RSO.size(), "Size mismatch!");
    my_assert(_Nkeep == _haloIDs.size(), "Size mismatch!");

    // log the class arrays in the memory log, now that their sizes are final
    memory.add_entry("SO x position", _XSO);
    memory.add_entry("SO y position", _YSO);
    memory.add_entry("SO z position", _ZSO);
    memory.add_entry("SO radii", _RSO);
    memory.add_entry("SO halo IDs", _haloIDs);
  }

  /**
   * @brief Get the total number of halos in the catalogue.
   *
   * @return Total number of halos, including subhalos.
   */
  inline size_t number_of_halos() const { return _Nhalo; }

  /**
   * @brief Get the total number of halos that are retained
   *
   * Properties for retained SOs can be queried using the approperiate functions
   * and an index in the range [0, number_to_keep()].
   *
   * @return Number of SOs actually stored in the SOTable.
   */
  inline size_t number_to_keep() const { return _Nkeep; }

  /**
   * @brief Get the x position of an SO.
   *
   * @param index SO index in the range [0, number_to_keep()].
   * @return X position of that SO, in co-moving VR distance units (Mpc).
   */
  inline double XSO(const size_t index) const { return _XSO[index]; }

  /**
   * @brief Get the y position of an SO.
   *
   * @param index SO index in the range [0, number_to_keep()].
   * @return Y position of that SO, in co-moving VR distance units (Mpc).
   */
  inline double YSO(const size_t index) const { return _YSO[index]; }

  /**
   * @brief Get the z position of an SO.
   *
   * @param index SO index in the range [0, number_to_keep()].
   * @return Z position of that SO, in co-moving VR distance units (Mpc).
   */
  inline double ZSO(const size_t index) const { return _ZSO[index]; }

  /**
   * @brief Get the radius of an SO.
   *
   * What "radius" means is determined by the constructor argument
   * "radius_selection_name"; see constructor documentation.
   *
   * @param index SO index in the range [0, number_to_keep()].
   * @return Radius of that SO, in co-moving VR distance units (Mpc).
   */
  inline double RSO(const size_t index) const { return _RSO[index]; }

  /**
   * @brief Get the halo ID for an SO.
   *
   * The halo ID is the unique ID assigned to the SO halo by VR, is independent
   * of the SO index, and is consistent with the VR catalogue.
   *
   * @param index SO index in the range [0, number_to_keep()].
   * @return Halo ID for the SO.
   */
  inline size_t haloID(const size_t index) const { return _haloIDs[index]; }
};

/**
 * @brief Auxiliary struct to keep track of orphan particles.
 *
 * Orphan particles are particles that have drifted beyond their top-level cell
 * and therefore cannot be correctly linked to an SO based on cell masking
 * alone.
 *
 * To not miss these particles in the reduced snapshot, we store all orphans
 * (there usually are few) and do a brute-force search through the SOs to find
 * SOs they belong to.
 */
struct Orphan {
  /*! @brief Particle type of the orphan. */
  int32_t type;
  /*! @brief Index of the particle within the file. If we read any of the
   *  arrays in the PartTypeX (with X = type) group, the properties of the
   *  orphan particle will be stored on that location. */
  int64_t index;
  /*! @brief Top-level cell the particle was (incorrectly) assigned to. We do
   *  not correct SWIFT's mistakes and keep treating the particle as if it
   *  belongs to that cell. */
  size_t cell;
  /*! @brief Position of the orphan particle. Really our only way to figure out
   *  where the orphan is located. */
  double position[3];
};

/**
 * @brief Main program entry point.
 *
 * The program proceeds in three phases:
 *  1. Halo catalogue parsing: all relevant properties from the halo catalogue
 *     are read in and stored in memory.
 *  2. SWIFT snapshot file parsing: the SWIFT snapshot file(s) is (are) read
 *     in chuncks of top-level cells. For each chunk, a particle ID - halo ID
 *     dictionary is constructed from the catalogue and used to assign
 *     halo IDs to all particles. Particles that do not appear in the catalogue
 *     get assigned a halo index -1.
 *  3. Output snapshot writing: a copy of the SWIFT snapshot(s) is made that
 *     contains exactly the same contents as the original snapshot, but with
 *     all particles with a halo index -1 filtered out.
 *
 * If the SWIFT snapshot is distributed among multiple files, then steps 2 and
 * 3 are performed separately for each file, and an additional step 3 is
 * performed in the end to also copy the virtual meta-file. In this case,
 * the program can be run over MPI, using at most the same number of ranks as
 * there are files (attempting to use more ranks will result in an error).
 * Each rank will still independently parse the halo catalogue (step 1).
 * Since the time needed to parse a single file is roughly constant, it is
 * strongly recommended to use a number of ranks that is a sensible divisor
 * of the number of files, to prevent ranks from idling.
 * Note that each rank will still try to use as many OpenMP threads as allowed
 * by OPENMP_NUM_THREADS. Allowing ranks to compete for threads is not
 * necessarily bad for performance, although pushing this too far does lead
 * to load imbalances between ranks that again cause ranks to be idle. So we
 * can only advise to choose the number of threads in a sensible way.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments. These are parsed positionally, nothing
 * fancy.
 */
int main(int argc, char **argv) {

  // first things first: initialise MPI
  MPI_Init(&argc, &argv);
  // and set our global variables (yuk!)
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);

  // start the global (yuk yuk!) timer
  global_timer.start();

  /// command line parsing

  if (argc < 8) {
    if (MPI_rank == 0) {
      std::cerr << "Usage: ./reduce_snapshots SOAP_OUTPUT_PREFIX "
                   "SNAPSHOT_FILE_PREFIX MEMBERSHIP__FILE_PREFIX "
                   "OUTPUT_FILE_PREFIX "
                   "RADIUS_SELECTION_NAME [CELLBUFSIZE] "
                   "[HDF5BUFSIZE]"
                << std::endl;
    }
    my_error("Wrong command line arguments!");
  }
  const std::string SOAP_output_prefix(argv[1]);
  const std::string snapshot_file_prefix(argv[2]);
  const std::string membership_file_prefix(argv[3]);
  const std::string output_file_prefix(argv[4]);
  std::string radius_selection_name(argv[5]);
  const uint32_t cellbufsize = (argc > 6) ? atoi(argv[6]) : 8;
  const size_t hdf5bufsize = (argc > 7) ? atoll(argv[7]) : -1;

  const std::string catalog_file = find_file(SOAP_output_prefix, "");
  //  const std::string siminfo_file = find_file(SOAP_output_prefix, ".siminfo");
  // it is perfectly normal for both the distributed and non-distributed snaphot
  // file to exist, so we disable only_one
  const std::string snapshot_file =
      find_file(snapshot_file_prefix, "", ".hdf5", 0, /* only_one = */ false);
  const std::string master_membership_file =
      find_file(membership_file_prefix, "", ".hdf5", 0);

  // output (only rank 0, because we don't want to clutter the stdout
  //  - just yet)
  if (MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL, "Running on %i rank(s).", MPI_size);
    timelog(LOGLEVEL_GENERAL, "Arguments:");
    timelog(LOGLEVEL_GENERAL, "SOAP output prefix: %s", SOAP_output_prefix.c_str());
    timelog(LOGLEVEL_GENERAL, "  Catalog file: %s", catalog_file.c_str());
    timelog(LOGLEVEL_GENERAL, "Snapshot file prefix: %s",
            snapshot_file_prefix.c_str());
    timelog(LOGLEVEL_GENERAL, "  Master snapshot: %s", snapshot_file.c_str());
    timelog(LOGLEVEL_GENERAL, "Membership file prefix: %s",
            membership_file_prefix.c_str());
    timelog(LOGLEVEL_GENERAL, "  Master membership file: %s",
            master_membership_file.c_str());
    timelog(LOGLEVEL_GENERAL, "Output file prefix: %s",
            output_file_prefix.c_str());
    timelog(LOGLEVEL_GENERAL, "Radius selection name: %s",
            radius_selection_name.c_str());
    timelog(LOGLEVEL_GENERAL, "Cellbufsize: %u", cellbufsize);
    timelog(LOGLEVEL_GENERAL, "HDF5bufsize: %s",
            human_readable_bytes(hdf5bufsize).c_str());
  }

  if (MPI_rank == 0) {
    fs::path output_file_path(output_file_prefix);
    auto parent_path = output_file_path.parent_path();
    if (!parent_path.empty()) {
      timelog(LOGLEVEL_GENERAL, "Output directory: %s", parent_path.c_str());
      if (!fs::exists(parent_path)) {
        timelog(LOGLEVEL_GENERAL, "Directory not found, creating it");
        fs::create_directories(parent_path);
      }
    }
  }

  // start HDF5
  // this is not an HDF5 requirement, but something our wrapper needs: we want
  // to disable default HDF5 error handling to make it easier to trace HDF5
  // errors within our code
  StartHDF5();
  // initialise the memory log
  MemoryLog memory_file(100);
  auto memory = memory_file.get_root();

  if (MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL, "Initialising SO catalog...");
  }

  // read the SOTable: this is the entire first step mentioned above
  const SOTable SOtable(SOAP_output_prefix, radius_selection_name, memory);

  MPI_Barrier(MPI_COMM_WORLD);

  if (MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL, "Done initialising catalog.");
    timelog(LOGLEVEL_GENERAL, "Reading master snapshot file...");
  }

  // for FLAMINGO, we really only have 4 particle types that are relevant
  // since we do not want to be general right now, we fix the number of types
  // to 5 and use this auxiliary array to convert our itype indices to the
  // actual type indices.
  const uint_fast32_t typelist[5] = {0, 1, 4, 5, 6};

  /// step 2 (from above)

  // first partfile read: determine how many files we have (and read some
  // constants while we are at it)
  HDF5FileOrGroup partfile = OpenFile(snapshot_file, HDF5FileModeRead);
  uint32_t numpart_total_lowword[7];
  uint32_t numpart_total_highword[7];
  int64_t numpart_total[7];
  int64_t new_numpart_total[7] = {0, 0, 0, 0, 0, 0, 0};
  double boxSize[3];
  int32_t nfile[1];
  HDF5FileOrGroup group = OpenGroup(partfile, "/Header");
  ReadArrayAttribute(group, "NumPart_Total", numpart_total_lowword);
  ReadArrayAttribute(group, "NumPart_Total_HighWord", numpart_total_highword);
  ReadArrayAttribute(group, "BoxSize", boxSize);
  ReadArrayAttribute(group, "NumFilesPerSnapshot", nfile);
  CloseGroup(group);

  // convert from Gadget2 limited size particle numbers to peta-scale and beyond
  for (uint_fast8_t i = 0; i < 7; ++i) {
    numpart_total[i] = numpart_total_highword[i];
    numpart_total[i] <<= 32;
    numpart_total[i] += numpart_total_lowword[i];
  }

  // read top-level cell meta-data
  group = OpenGroup(partfile, "/Cells");
  HDF5FileOrGroup subgroup = OpenGroup(group, "Meta-data");
  int32_t cell_dimension[3];
  double cell_size[3];
  ReadArrayAttribute(subgroup, "dimension", cell_dimension);
  ReadArrayAttribute(subgroup, "size", cell_size);
  CloseGroup(subgroup);

  const double maxcellsize = 0.5 * std::sqrt(cell_size[0] * cell_size[0] +
                                             cell_size[1] * cell_size[1] +
                                             cell_size[2] * cell_size[2]);

  const int32_t num_cell =
      cell_dimension[0] * cell_dimension[1] * cell_dimension[2];
  std::vector<double> cell_centres(3 * num_cell);
  ReadEntireDataset(group, "Centres", cell_centres);
  memory.add_entry("Cell centres", cell_centres);

  if (num_cell % cellbufsize > 0) {
    my_error("Cell buffer size (%u) needs to be multiple of cell number (%i)!",
             cellbufsize, num_cell);
  }

  subgroup = OpenGroup(group, "OffsetsInFile");
  std::vector<int64_t> part_offsets[5];
  // we will be updating the offsets (and sizes) when we remove particles
  // it (sort of?) makes sense to create these vectors now
  std::vector<int64_t> new_offsets[5];
  for (uint_fast32_t itype = 0; itype < 5; ++itype) {
    if (numpart_total[typelist[itype]] == 0) {
      continue;
    }
    part_offsets[itype].resize(num_cell);
    new_offsets[itype].resize(num_cell, 0);
    std::ostringstream groupnamestr;
    groupnamestr << "PartType" << typelist[itype];
    ReadEntireDataset(subgroup, groupnamestr.str(), part_offsets[itype]);
    memory.add_entry("Part offsets " + groupnamestr.str(), part_offsets[itype]);
    memory.add_entry("New offsets " + groupnamestr.str(), new_offsets[itype]);
  }
  CloseGroup(subgroup);

  subgroup = OpenGroup(group, "Counts");
  std::vector<int64_t> part_sizes[5];
  std::vector<int64_t> new_sizes[5];
  for (uint_fast32_t itype = 0; itype < 5; ++itype) {
    if (numpart_total[typelist[itype]] == 0) {
      continue;
    }
    part_sizes[itype].resize(num_cell);
    new_sizes[itype].resize(num_cell, 0);
    std::ostringstream groupnamestr;
    groupnamestr << "PartType" << typelist[itype];
    ReadEntireDataset(subgroup, groupnamestr.str(), part_sizes[itype]);
    memory.add_entry("Part sizes " + groupnamestr.str(), part_sizes[itype]);
    memory.add_entry("New sizes " + groupnamestr.str(), new_sizes[itype]);
  }
  CloseGroup(subgroup);

  std::vector<int32_t> files[5];
  // note: no need to update this vector: particles stay in the file they are in
  // if they are allowed to stay at all
  if (nfile[0] > 1) {
    subgroup = OpenGroup(group, "Files");
    for (uint_fast32_t itype = 0; itype < 5; ++itype) {
      if (numpart_total[typelist[itype]] == 0) {
        continue;
      }
      files[itype].resize(num_cell);
      std::ostringstream groupnamestr;
      groupnamestr << "PartType" << typelist[itype];
      ReadEntireDataset(subgroup, groupnamestr.str(), files[itype]);
      memory.add_entry("Part files " + groupnamestr.str(), files[itype]);
    }
    CloseGroup(subgroup);
  }
  CloseGroup(group);

  // we are done reading the particle file (for now), so we close it
  // rank 0 will open it again very soon.
  CloseFile(partfile);

  MPI_Barrier(MPI_COMM_WORLD);
  // this output is actually still useful
  if (MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL, "Done reading master snapshot file.");
    timelog(LOGLEVEL_GENERAL, "BoxSize: %g %g %g", boxSize[0], boxSize[1],
            boxSize[2]);
    timelog(LOGLEVEL_GENERAL, "Number of files: %i", nfile[0]);
    timelog(LOGLEVEL_GENERAL, "Number of cells: %i x %i x %i = %i",
            cell_dimension[0], cell_dimension[1], cell_dimension[2], num_cell);
    timelog(LOGLEVEL_GENERAL, "Distributing files across ranks...");
  }

  // distribute files over processes
  // nothing fancy here: if there are N files per rank, then rank 0 gets the
  // first N (0..N-1), rank 1 gets N..2N-1 and so on
  std::vector<std::pair<int, std::string>> local_files;
  std::vector<std::string> all_files(nfile[0]);
  // prevent us from deliberately wasting CPU time by having idle ranks
  if (MPI_size > nfile[0]) {
    my_error("Using more ranks than there are files, this is pointless!");
  }
  // it would be good if MPI_size was a proper divisor of nfile[0], but
  // we make sure each rank (except the last one) gets one more file if it is
  // not
  const int files_per_rank = nfile[0] / MPI_size + (nfile[0] % MPI_size > 0);
  for (int ifile = 0; ifile < nfile[0]; ++ifile) {
    // we already compose a list of all output files
    // rank 0 will need this to write the virtual meta-file in the end
    all_files[ifile] =
        compose_filename(output_file_prefix, ".hdf5", ifile, nfile[0] > 1);
    if (ifile / files_per_rank == MPI_rank) {
      local_files.push_back(
          std::make_pair(ifile, find_file(snapshot_file_prefix, "", ".hdf5",
                                          ifile, /* only_one = */ false)));
    }
  }
  if (MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL, "Done distributing files.");
    timelog(LOGLEVEL_GENERAL, "Number of files per rank: %i", files_per_rank);
  }
  // definitely useful output. We even stagger the ranks, so that the messages
  // don't get mixed up
  for (int irank = 0; irank < MPI_size; ++irank) {
    if (irank == MPI_rank) {
      timelog(LOGLEVEL_GENERAL, "Local files on rank %i:", MPI_rank);
      for (size_t ifile = 0; ifile < local_files.size(); ++ifile) {
        timelog(LOGLEVEL_GENERAL, "  %i %s", local_files[ifile].first,
                local_files[ifile].second.c_str());
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  memory.add_entry("Local files", local_files);

  if (MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL, "Starting file loop...");
  }

  // we need to store the number of (retained) particles in each new output
  // file, since this determines the new offsets of the top-level cells in the
  // new virtual meta-file
  std::vector<int64_t> file_offsets[5];
  for (uint_fast8_t itype = 0; itype < 5; ++itype) {
    file_offsets[itype].resize(nfile[0], 0);
  }
  // start the main file loop
  // this combines steps 2 and 3 for individual distributed input files
  auto memory_ifile_loop = memory.add_loop_scope("File loop");
  for (size_t ifile = 0; ifile < local_files.size(); ++ifile) {
    std::stringstream ifilename;
    ifilename << "File" << ifile;
    auto memory_ifile = memory_ifile_loop.add_block(ifilename.str());

    timelog(LOGLEVEL_GENERAL, "Reading file %s",
            local_files[ifile].second.c_str());

    // (re)open the file and read the file specific meta-data
    partfile = OpenFile(local_files[ifile].second, HDF5FileModeRead);
    int32_t tfile[1];
    int64_t numpart_thisfile[7];
    group = OpenGroup(partfile, "/Header");
    ReadArrayAttribute(group, "ThisFile", tfile);
    ReadArrayAttribute(group, "NumPart_ThisFile", numpart_thisfile);
    CloseGroup(group);

    // we will store the new halo ID dataset for the entire file
    // we also need to store the orphans on the file level, since there is no
    // way (per definition) to know which top-level cell an orphan belongs to
    size_t tot_SO_count = 0;
    std::vector<struct Orphan> orphans;
    std::vector<int64_t> haloIDs[5];
    haloIDs[0].resize(numpart_thisfile[0], -1);
    haloIDs[1].resize(numpart_thisfile[1], -1);
    haloIDs[2].resize(numpart_thisfile[4], -1);
    haloIDs[3].resize(numpart_thisfile[5], -1);
    haloIDs[4].resize(numpart_thisfile[6], -1);
    memory_ifile.add_entry(ifilename.str() + " haloIDs0", haloIDs[0]);
    memory_ifile.add_entry(ifilename.str() + " haloIDs1", haloIDs[1]);
    memory_ifile.add_entry(ifilename.str() + " haloIDs2", haloIDs[2]);
    memory_ifile.add_entry(ifilename.str() + " haloIDs3", haloIDs[3]);
    memory_ifile.add_entry(ifilename.str() + " haloIDs4", haloIDs[4]);
    // count
    //  - the number of retained particles
    //  - the number of particles outside the box (for analysis)
    //  - the number of duplicates, i.e. particles that belong to multiple SOs
    size_t keepcount = 0;
    size_t outcount = 0;
    size_t dupcount = 0;
    size_t filtercount = 0;
    // allocate the cell_SOs vector once and then reuse it
    // assuming the size does not change much between iterations, this saves
    // on reallocations
    std::vector<size_t> cell_SOs;
    auto memory_icellchunk_loop = memory_ifile.add_loop_scope("CellChunk loop");
    // now process the file in chunks of top-level cells
    // reading chunks instead of all cells saves on memory and speeds up the
    // construction of the ID table
    for (uint32_t icellchunk = 0; icellchunk < num_cell / cellbufsize;
         ++icellchunk) {
      std::stringstream icellchunkname;
      icellchunkname << ifilename.str() << " CellChunk" << icellchunk;
      auto memory_icellchunk =
          memory_icellchunk_loop.add_block(icellchunkname.str());
      timelog(LOGLEVEL_CHUNKLOOPS, "Parsing cell chunk %i of %i", icellchunk,
              num_cell / cellbufsize);
      timelog(LOGLEVEL_CHUNKLOOPS, "Counting overlapping SOs...");
      // clear the previous list of SOs that overlap
      // clear() does not (necessarily) reallocate, so the memory from the
      // previous iteration will (hopefully) be used again
      cell_SOs.clear();
      // count the number of SOs that (possibly) overlap with the cells in this
      // chunk
      // we keep track of the overlap for every cell separately, so SOs that
      // overlap with multiple cells will appear multiple times
      // we also keep track of the start and end of each cell in the SO list,
      // so that we can iterate over the SOs for a specific cell later on
      std::vector<size_t> SO_overlap_offsets(cellbufsize + 1, 0);
      for (uint32_t ichunk = 0; ichunk < cellbufsize; ++ichunk) {
        // store the offset of this cell in the overlapping SO list
        SO_overlap_offsets[ichunk] = cell_SOs.size();
        const size_t icell = cellbufsize * icellchunk + ichunk;
        if (nfile[0] > 1) {
          bool has_parts = false;
          for (int_fast8_t itype = 0; itype < 5; ++itype) {
            if (files[itype].size() > 0) {
              has_parts |= files[itype][icell] == tfile[0];
            }
          }
          // make sure this file has particles in the cell
          if (!has_parts) {
            continue;
          }
        }
        // now check for overlapping SOs
        for (size_t iSO = 0; iSO < SOtable.number_to_keep(); ++iSO) {
          // we determine the shortest distance between the top-level cell
          // centre and the centre of the SO, among all periodic copies of the
          // SO
          double dx[3] = {SOtable.XSO(iSO) - cell_centres[3 * icell],
                          SOtable.YSO(iSO) - cell_centres[3 * icell + 1],
                          SOtable.ZSO(iSO) - cell_centres[3 * icell + 2]};
          if (dx[0] < -0.5 * boxSize[0]) {
            dx[0] += boxSize[0];
          }
          if (dx[0] >= 0.5 * boxSize[0]) {
            dx[0] -= boxSize[0];
          }
          if (dx[1] < -0.5 * boxSize[1]) {
            dx[1] += boxSize[1];
          }
          if (dx[1] >= 0.5 * boxSize[1]) {
            dx[1] -= boxSize[1];
          }
          if (dx[2] < -0.5 * boxSize[2]) {
            dx[2] += boxSize[2];
          }
          if (dx[2] >= 0.5 * boxSize[2]) {
            dx[2] -= boxSize[2];
          }
          const double d =
              std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
          // we use a very conservative estimate: the SO is assumed to overlap
          // if the distance is smaller than the furthest possible distance
          // within the cell (segment from cell centre to one of the cell
          // corners) plus the full radius of the SO
          if (d < maxcellsize + SOtable.RSO(iSO)) {
            cell_SOs.push_back(iSO);
          }
        }
      }
      // set the beyond end index in the overlapping SO cell list, so that we
      // can generally retrieve the SOs that overlap with cell i by indexing
      // cell_SOs[SO_overlap_offsets[i]:SO_overlap_offsets[i+1]]
      SO_overlap_offsets[cellbufsize] = cell_SOs.size();
      // not sure why we are logging this; since there are very likely overlaps
      // between different chunks, this number is quite meaningless
      tot_SO_count += cell_SOs.size();
      timelog(LOGLEVEL_CHUNKLOOPS, "Constructing chunk ID table...");
      memory_icellchunk.add_entry(icellchunkname.str() + " cell SOs", cell_SOs);
      timelog(LOGLEVEL_CHUNKLOOPS, "Reading particles...");
      auto memory_itype_loop =
          memory_icellchunk.add_loop_scope("PartType loop");
      // now process the particles in the chunk
      // we have to process the different particle types separately
      for (uint_fast32_t itype = 0; itype < 5; ++itype) {
        if (numpart_thisfile[typelist[itype]] == 0) {
          continue;
        }
        // compose the particle group name (PartType0/1/4/5)
        std::ostringstream groupnamestr;
        groupnamestr << "PartType" << typelist[itype];
        std::stringstream itypename;
        itypename << icellchunkname.str() << " " << groupnamestr.str();
        auto memory_itype = memory_itype_loop.add_block(itypename.str());
        timelog(LOGLEVEL_CHUNKLOOPS, "%s", groupnamestr.str().c_str());
        // open the group in the HDF5 file
        group = OpenGroup(partfile, groupnamestr.str());
        size_t totsize = 0;
        timelog(LOGLEVEL_CHUNKLOOPS, "Selecting particle chunks...");
        // figure out which parts of the particle datasets belong to top-level
        // cells in the chunk we are processing
        std::vector<struct HDF5Chunk> chunks;
        // we need two auxiliary arrays:
        //  - an index array for argsorting the chunks
        //  - an array that links the nominal cell index in the chunk (that
        //    assumes all cells in the chunk are present within this distributed
        //    file) to the actual cell index (that takes into account missing
        //    cells)
        std::vector<size_t> indices;
        std::vector<uint32_t> ichunks;
        // second index that only counts cells that are actually present in this
        // distributed file
        size_t idx = 0;
        for (uint32_t ichunk = 0; ichunk < cellbufsize; ++ichunk) {
          const size_t icell = cellbufsize * icellchunk + ichunk;
          if (nfile[0] == 1 || files[itype][icell] == tfile[0]) {
            chunks.resize(chunks.size() + 1);
            chunks[idx].offset = part_offsets[itype][icell];
            chunks[idx].size = part_sizes[itype][icell];
            totsize += chunks[idx].size;
            indices.resize(indices.size() + 1);
            indices[idx] = idx;
            ichunks.resize(ichunks.size() + 1);
            ichunks[idx] = ichunk;
            ++idx;
          }
        }
        memory_itype.add_entry(itypename.str() + " chunks", chunks);
        memory_itype.add_entry(itypename.str() + " indices", indices);
        memory_itype.add_entry(itypename.str() + " ichunks", ichunks);
        // in the distributed case, it is very likely that there will be no
        // particles of this type that overlap with this chunk, so we can
        // continue with the next particle type
        if (chunks.size() == 0) {
          timelog(LOGLEVEL_CHUNKLOOPS, "Nothing to read here!");
          continue;
        }
        // the chunks we created are randomly ordered
        // when we read their data from the HDF5 file, they will be ordered
        // according to the chunk offset
        // to recover the correct particles for each cell in the chunk, we need
        // to argsort the chunks based on the chunk offset
        std::sort(indices.begin(), indices.end(),
                  [&chunks](const size_t a, const size_t b) {
                    return chunks[a].offset < chunks[b].offset;
                  });
        // now read the chunks from the snapshot file
        std::vector<double> partpos(3 * totsize);
        memory_itype.add_entry(itypename.str() + " partpos", partpos);
        timelog(LOGLEVEL_CHUNKLOOPS, "Reading chunks...");
        ReadPartial3DDataset(group, "Coordinates", chunks, partpos);
        timelog(LOGLEVEL_CHUNKLOOPS, "Assigning SO IDs to particles...");
        // loop over the (present) top-level cells in the chunk
        size_t poffset = 0;
        for (uint32_t ic = 0; ic < chunks.size(); ++ic) {
          // recover the unsorted cell index
          const uint32_t ichunk = indices[ic];
          // recover the nominal cell index
          const size_t icell = cellbufsize * icellchunk + ichunks[ichunk];
          // get the start and end of the SOs that overlap with this cell in
          // the list of all overlapping SOs
          const size_t SO_overlap_offset = SO_overlap_offsets[ichunks[ichunk]];
          const size_t SO_overlap_size =
              SO_overlap_offsets[ichunks[ichunk] + 1];
          // get the bounding box of the cell (for orphan detection)
          double cell_low[3] = {
              cell_centres[3 * icell] - 0.5 * cell_size[0],
              cell_centres[3 * icell + 1] - 0.5 * cell_size[1],
              cell_centres[3 * icell + 2] - 0.5 * cell_size[2]};
          double cell_high[3] = {
              cell_centres[3 * icell] + 0.5 * cell_size[0],
              cell_centres[3 * icell + 1] + 0.5 * cell_size[1],
              cell_centres[3 * icell + 2] + 0.5 * cell_size[2]};
          // deal with the periodic boundaries
          // we make sure all cell boundaries are within the periodic box
          // and we record which boundaries were wrapped, since this changes
          // the condition we need to check whether a particle lies in the cell
          bool wrap[3] = {false, false, false};
          for (uint_fast8_t ix = 0; ix < 3; ++ix) {
            if (cell_low[ix] < 0.) {
              cell_low[ix] += boxSize[ix];
              wrap[ix] = true;
            }
            if (cell_high[ix] >= boxSize[ix]) {
              cell_high[ix] -= boxSize[ix];
              wrap[ix] = true;
            }
          }
          // get the offset and size of this cell's particles within the
          // particle arrays
          const int64_t offset = part_offsets[itype][icell];
          const int64_t size = part_sizes[itype][icell];
          my_assert(size == static_cast<int64_t>(chunks[ichunk].size),
                    "Size mismatch!");
          // the particles are processed using multiple threads
          // each thread uses its own accumulators
          size_t this_keepcount = 0;
          size_t this_outcount = 0;
          size_t this_dupcount = 0;
          size_t this_filtercount = 0;
#pragma omp parallel for default(shared) \
  reduction(+ : this_keepcount, this_outcount, this_dupcount, this_filtercount)
          for (int64_t ipart = 0; ipart < size; ++ipart) {
            // since we only read the chunks of the particle arrays that are
            // actually relevant, the offset stored in the cell meta-data does
            // not actually match the offset in our particle arrays
            // the actual offset is given by the number of particles in all
            // the present top-level cells in the chunk that were already
            // parsed before, in the read order (so the order in which we
            // argsorted the cells)
            // we keep a running poffset counter that is updated with the cell
            // size after every iteration
            const int64_t totipart = poffset + ipart;
            // check if the particle is in the box and in the cell
            bool is_out = false;
            bool out_of_box = false;
            for (uint_fast8_t ix = 0; ix < 3; ++ix) {
              // if the cell positions were periodically wrapped, we use a
              // different condition to check whether the particle is in the
              // cell
              if (!wrap[ix]) {
                if (cell_low[ix] > partpos[3 * totipart + ix] ||
                    cell_high[ix] < partpos[3 * totipart + ix]) {
                  is_out = true;
                }
              } else {
                if (cell_low[ix] > partpos[3 * totipart + ix] &&
                    cell_high[ix] < partpos[3 * totipart + ix]) {
                  is_out = true;
                }
              }
              if (partpos[3 * totipart + ix] < 0.) {
                out_of_box = true;
                // display a message for each particle that is detected to be
                // outside the box
#pragma omp critical
                timelog(LOGLEVEL_GENERAL, "Particle outside box: %g %g %g!",
                        partpos[3 * totipart], partpos[3 * totipart + 1],
                        partpos[3 * totipart + 2]);
                partpos[3 * totipart + ix] += boxSize[ix];
              }
              if (partpos[3 * totipart + ix] >= boxSize[ix]) {
                out_of_box = true;
#pragma omp critical
                timelog(LOGLEVEL_GENERAL, "Particle outside box: %g %g %g!",
                        partpos[3 * totipart], partpos[3 * totipart + 1],
                        partpos[3 * totipart + 2]);
                partpos[3 * totipart + ix] -= boxSize[ix];
              }
            }
            // count the number of particles outside the box
            this_outcount += out_of_box;
            if (is_out) {
              // if the particle is outside the cell, then our list of SO
              // candidates might be wrong for this particle (since the SO
              // selection was based on the assumption that all particles
              // lie within the cell)
              // we have to record this particle as an orphan and process it
              // on a more general level
              struct Orphan orphan;
              orphan.type = itype;
              // in this case, the index of the particle is w.r.t. the total
              // particle list, not the one that only contains particles within
              // this cell chunk
              orphan.index = offset + ipart;
              orphan.cell = icell;
              orphan.position[0] = partpos[3 * totipart];
              orphan.position[1] = partpos[3 * totipart + 1];
              orphan.position[2] = partpos[3 * totipart + 2];
              // we are multi-threading, so we have to make sure only one thread
              // is allowed to update the (shared) orphan vector at a time
#pragma omp critical
              orphans.push_back(orphan);
            } else {
              for (uint_fast32_t iSO = SO_overlap_offset; iSO < SO_overlap_size;
                   ++iSO) {
                const int64_t SOID = cell_SOs[iSO];
                double dx = partpos[3 * totipart] - SOtable.XSO(SOID);
                if (dx < -0.5 * boxSize[0])
                  dx += boxSize[0];
                if (dx >= 0.5 * boxSize[0])
                  dx -= boxSize[0];
                double dy = partpos[3 * totipart + 1] - SOtable.YSO(SOID);
                if (dy < -0.5 * boxSize[1])
                  dy += boxSize[1];
                if (dy >= 0.5 * boxSize[1])
                  dy -= boxSize[1];
                double dz = partpos[3 * totipart + 2] - SOtable.ZSO(SOID);
                if (dz < -0.5 * boxSize[2])
                  dz += boxSize[2];
                if (dz >= 0.5 * boxSize[2])
                  dz -= boxSize[2];
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                // filter on radius
                if (d <= SOtable.RSO(SOID)) {
                  // check if we already have a match for this particle
                  // note that we do not need to worry about thread-safety
                  // here, since only one thread will process this particular
                  // particle
                  // also note that we use the global offset here and not
                  // the reduced one, since the haloIDs array covers all
                  // particles in the file
                  if (haloIDs[itype][offset + ipart] == -1) {
                    // normal case: assign this SO to the particle
                    haloIDs[itype][offset + ipart] = SOID;
                    ++this_keepcount;
                  } else {
                    // duplicate! We just keep the first halo id that was
                    // assigned to the particle.
                    ++this_dupcount;
                  }
                }
              }
            }
          }
          // update the (local) offset
          poffset += size;
          // update the global counters outside the thread-parallel region
          keepcount += this_keepcount;
          outcount += this_outcount;
          dupcount += this_dupcount;
          filtercount += this_filtercount;
          my_assert(new_sizes[itype][icell] == 0, "Overwriting cell size!");
          // register the new number of particles of this type within the top-
          // level cell
          // note that this number might still get updated once we process the
          // orphans
          new_sizes[itype][icell] = this_keepcount;
        }
        // we are done with this type, close the corresponding group
        CloseGroup(group);
      }
    }
    // some output (was useful at some point)
    timelog(LOGLEVEL_CHUNKLOOPS, "Chunk stats:");
    timelog(LOGLEVEL_CHUNKLOOPS, "  total SO count: %zu, SOs to keep: %zu",
            tot_SO_count, SOtable.number_to_keep());
    timelog(LOGLEVEL_CHUNKLOOPS,
            "  particles to keep in this chunk (excluding orphans): %zu",
            keepcount);
    timelog(LOGLEVEL_CHUNKLOOPS, "  particles in this chunk outside box: %zu",
            outcount);
    timelog(LOGLEVEL_CHUNKLOOPS,
            "  particles in this chunk belonging to multiple SOs: %zu",
            dupcount);
    timelog(LOGLEVEL_CHUNKLOOPS, "  number of orphans in this chunk: %zu",
            orphans.size());
    timelog(LOGLEVEL_CHUNKLOOPS,
            "  number of particles filtered out by radius: %zu", filtercount);

    memory_ifile.add_entry(ifilename.str() + " Orphans", orphans);

    // we are done processing the regular particles that are within their
    // top-level cells
    // now we process the orphans, particles that have drifted outside their
    // cell
    size_t this_keepcount = 0;
#pragma omp parallel for default(shared) reduction(+ : this_keepcount)
    for (size_t i = 0; i < orphans.size(); ++i) {
      const struct Orphan &orphan = orphans[i];
      // since we have no way of spatially filtering the SOs for orphans, we
      // need to brute-force search through all of them
      for (size_t iSO = 0; iSO < SOtable.number_to_keep(); ++iSO) {
        // like in the normal case, we compute the distance between the particle
        // and the SO centre and perform a radius check
        double dx = orphan.position[0] - SOtable.XSO(iSO);
        if (dx < -0.5 * boxSize[0])
          dx += boxSize[0];
        if (dx >= 0.5 * boxSize[0])
          dx -= boxSize[0];
        double dy = orphan.position[1] - SOtable.YSO(iSO);
        if (dy < -0.5 * boxSize[1])
          dy += boxSize[1];
        if (dy >= 0.5 * boxSize[1])
          dy -= boxSize[1];
        double dz = orphan.position[2] - SOtable.ZSO(iSO);
        if (dz < -0.5 * boxSize[2])
          dz += boxSize[2];
        if (dz >= 0.5 * boxSize[2])
          dz -= boxSize[2];
        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (d <= SOtable.RSO(iSO)) {
          // again, we need to deal with duplicates
          // accessing the haloIDs on orphan index is thread-safe, since each
          // orphan is only processed once by one thread
          if (haloIDs[orphan.type][orphan.index] == -1) {
            haloIDs[orphan.type][orphan.index] = iSO;
            // update the counter for the cell; it contains another (retained)
            // particle
            // this is not thread safe, since multiple orphans might share the
            // same cell
#pragma omp critical
            ++new_sizes[orphan.type][orphan.cell];
            ++this_keepcount;
          }
        }
      }
    }
    // update the number of retained particles for the cell with orphans that
    // were retained
    keepcount += this_keepcount;
    timelog(LOGLEVEL_CHUNKLOOPS,
            "  particles to keep in this chunk (including orphans): %zu",
            keepcount);

    // now we are absolutely done with all particles in the file, we can count
    // the number of particles that remain
    size_t totcount = 0;
    size_t oldtotcount = 0;
    size_t SOpcount[5] = {0, 0, 0, 0, 0};
    timelog(LOGLEVEL_CHUNKLOOPS, "Remaining particles in chunk:");
    for (uint_fast32_t itype = 0; itype < 5; ++itype) {
      size_t this_SOpcount = 0;
#pragma omp parallel for default(shared) reduction(+ : this_SOpcount)
      for (size_t ipart = 0; ipart < haloIDs[itype].size(); ++ipart) {
        this_SOpcount += (haloIDs[itype][ipart] >= 0);
      }
      totcount += this_SOpcount;
      oldtotcount += haloIDs[itype].size();
      SOpcount[itype] = this_SOpcount;
      timelog(LOGLEVEL_CHUNKLOOPS, "  PartType%" PRIuFAST32 ": %zu of %zu",
              typelist[itype], this_SOpcount, haloIDs[itype].size());
    }
    timelog(LOGLEVEL_CHUNKLOOPS, "  AllParticles: %zu of %zu", totcount,
            oldtotcount);
    my_assert(totcount == keepcount, "Particle number mismatch!");

    const int64_t new_numpart_thisfile[7] = {static_cast<int64_t>(SOpcount[0]),
                                             static_cast<int64_t>(SOpcount[1]),
                                             0,
                                             0,
                                             static_cast<int64_t>(SOpcount[2]),
                                             static_cast<int64_t>(SOpcount[3]),
                                             static_cast<int64_t>(SOpcount[4])};

    // now create the new output file that only contains the remaining particles
    const std::string output_file_name = compose_filename(
        output_file_prefix, ".hdf5", local_files[ifile].first, nfile[0] > 1);
#ifdef SET_STRIPING
    {
      std::stringstream stripe_command;
      stripe_command << "lfs setstripe -c 1 -i " << (MPI_rank % 192) << " "
                     << output_file_name;
      timelog(LOGLEVEL_GENERAL, "Running lfs command \"%s\".",
              stripe_command.str().c_str());
      int return_value = system(stripe_command.str().c_str());
      if (return_value != 0) {
        timelog(LOGLEVEL_GENERAL,
                "Unable to set striping for file \"%s\". Continuing anyway.",
                output_file_name.c_str());
      }
    }
#endif
    HDF5FileOrGroup outfile = OpenFile(output_file_name, HDF5FileModeWrite);

    // first, copy everything in the file that does not require changing
    // by using a very general copy method, we make sure that the program
    // works regardless of the file contents
    // instead of copying everything we want to copy (which requires detailed
    // knowledge of the file), we copy everything except the things we know have
    // changed (which requires a lot less knowledge)
    timelog(LOGLEVEL_GENERAL,
            "Copying existing particle data to new file %s...",
            output_file_name.c_str());
    std::vector<std::string> blacklist;
    blacklist.push_back("PartType0");
    blacklist.push_back("PartType1");
    blacklist.push_back("PartType4");
    blacklist.push_back("PartType5");
    blacklist.push_back("PartType6");
    CopyEverythingExcept(partfile, outfile, blacklist);

    // we have not copied the particle data, since these datasets actually
    // change
    // loop over the particle types and manually copy the retained data over
    timelog(LOGLEVEL_GENERAL, "Adding halo ID dataset...");
    auto memory_itype_loop = memory_ifile.add_loop_scope("PartType loop");
    for (uint_fast32_t itype = 0; itype < 5; ++itype) {
      // only bother if there are still particles left to copy
      if (new_numpart_thisfile[typelist[itype]] == 0) {
        continue;
      }
      // compose the group name (PartType0/1/4/5/6)
      std::ostringstream groupnamestr;
      groupnamestr << "PartType" << typelist[itype];
      std::stringstream itypename;
      itypename << ifilename.str() << " " << groupnamestr.str();
      auto memory_itype = memory_itype_loop.add_block(itypename.str());
      timelog(LOGLEVEL_GENERAL, "%s", groupnamestr.str().c_str());
      // copy the group with all its attributes but without any of the datasets
      CopyGroupWithoutData(partfile, outfile, groupnamestr.str());
      // update the number of particles in the group
      const HDF5FileOrGroup new_group = OpenGroup(outfile, groupnamestr.str());
      const int64_t new_numpart[1] = {static_cast<int64_t>(SOpcount[itype])};
      ReplaceArrayAttribute(new_group, "NumberOfParticles", new_numpart);
      // and the number of fields (if it exists), since we add a field
      if (AttributeExists(new_group, "NumberOfFields")) {
        int32_t num_fields[1];
        ReadArrayAttribute(new_group, "NumberOfFields", num_fields);
        ++num_fields[0];
        ReplaceArrayAttribute(new_group, "NumberOfFields", num_fields);
      }
      // now open the old group, since we will be copying over stuff from it
      const HDF5FileOrGroup old_group = OpenGroup(partfile, groupnamestr.str());
      timelog(LOGLEVEL_GENERAL, "Creating mask...");
      // mask out all particles that need copying
      // we also construct the masked halo ID array, which we will add as a new
      // dataset
      std::vector<int64_t> hID(SOpcount[itype]);
      std::vector<bool> mask(haloIDs[itype].size(), false);
      size_t iSO = 0;
      for (size_t i = 0; i < haloIDs[itype].size(); ++i) {
        if (haloIDs[itype][i] >= 0) {
          hID[iSO] = SOtable.haloID(haloIDs[itype][i]);
          mask[i] = true;
          ++iSO;
        }
      }
      memory_itype.add_entry("hID", hID);
      memory_itype.add_entry("mask", mask);
      timelog(LOGLEVEL_GENERAL, "Writing new dataset...");
      // write the new halo ID dataset and all its attributes
      WriteDataset(new_group, "SOIDs", hID);
      HDF5FileOrGroup ds = OpenDataset(new_group, "SOIDs");
      const float float_zero = 0.0f;
      const double double_zero = 0.;
      AddAttribute(ds, "U_M exponent", float_zero);
      AddAttribute(ds, "U_L exponent", float_zero);
      AddAttribute(ds, "U_t exponent", float_zero);
      AddAttribute(ds, "U_I exponent", float_zero);
      AddAttribute(ds, "U_T exponent", float_zero);
      AddAttribute(ds, "h-scale exponent", float_zero);
      AddAttribute(ds, "a-scale exponent", float_zero);
      AddAttribute(ds, "Expression for physical CGS units", "[ - ] ");
      AddAttribute(ds, "Lossy compression filter", "None");
      AddAttribute(
          ds,
          "Conversion factor to CGS (not including cosmological corrections)",
          double_zero);
      AddAttribute(ds,
                   "Conversion factor to physical CGS (including cosmological "
                   "corrections)",
                   double_zero);
      AddAttribute(ds, "Description",
                   "ID of the most massive spherical overdensity the particles "
                   "belong to");
      CloseDataset(ds);
      timelog(LOGLEVEL_GENERAL, "Copying membership datasets...");
      // open the membership file
      const std::string memberfilename = find_file(
          membership_file_prefix, "", ".hdf5", local_files[ifile].first, false);
      HDF5FileOrGroup memberfile = OpenFile(memberfilename, HDF5FileModeRead);
      HDF5FileOrGroup membergroup = OpenGroup(memberfile, groupnamestr.str());
      std::vector<std::string> membernames;
      std::vector<std::string> memberdescr;
      membernames.push_back("GroupNr_all");
      memberdescr.push_back("Index of halo in which this particle is a member "
                            "(bound or unbound), or -1 if none");
      membernames.push_back("GroupNr_bound");
      memberdescr.push_back("Index of halo in which this particle is a bound "
                            "member, or -1 if none");
      membernames.push_back("Rank_bound");
      memberdescr.push_back("Ranking by binding energy of the bound particles "
                            "(first in halo=0), or -1 if not bound");
      for (uint_fast8_t imember = 0; imember < 3; ++imember) {
        std::vector<int32_t> memberdata(mask.size());
        std::vector<int32_t> memberdata_masked(SOpcount[itype]);
        ReadEntireDataset(membergroup, membernames[imember], memberdata);
        // mask out the elements we want to keep
        iSO = 0;
        for (size_t ihalo = 0; ihalo < mask.size(); ++ihalo) {
          if (mask[ihalo]) {
            memberdata_masked[iSO] = memberdata[ihalo];
            ++iSO;
          }
        }
        WriteDataset(new_group, membernames[imember], memberdata_masked);
        HDF5FileOrGroup memberds = OpenDataset(new_group, membernames[imember]);
        AddAttribute(memberds, "U_M exponent", float_zero);
        AddAttribute(memberds, "U_L exponent", float_zero);
        AddAttribute(memberds, "U_t exponent", float_zero);
        AddAttribute(memberds, "U_I exponent", float_zero);
        AddAttribute(memberds, "U_T exponent", float_zero);
        AddAttribute(memberds, "h-scale exponent", float_zero);
        AddAttribute(memberds, "a-scale exponent", float_zero);
        AddAttribute(memberds, "Expression for physical CGS units", "[ - ] ");
        AddAttribute(memberds, "Lossy compression filter", "None");
        AddAttribute(
            memberds,
            "Conversion factor to CGS (not including cosmological corrections)",
            double_zero);
        AddAttribute(
            memberds,
            "Conversion factor to physical CGS (including cosmological "
            "corrections)",
            double_zero);
        AddAttribute(memberds, "Description", memberdescr[imember].c_str());
        CloseDataset(memberds);
      }
      CloseGroup(membergroup);
      CloseFile(memberfile);
      timelog(LOGLEVEL_GENERAL, "Copying other datasets...");
      // now copy over the other datasets
      // again, we use a very general, dataset agnostic method, to make sure
      // all data are copied, regardless of the file layout
      const size_t copysize = CopyDatasets(old_group, new_group,
                                           SOpcount[itype], mask, hdf5bufsize);
      memory_itype.add_entry("Dataset copy", copysize);
      // close the groups, we are done with them
      CloseGroup(new_group);
      CloseGroup(old_group);
      // now figure out the new top-level cell offsets for the reduced file
      // since the top-level cells are not necessarily sorted, we first have
      // to argsort the top-level cells based on their old offset
      std::vector<size_t> indices(new_sizes[itype].size(), 0);
      memory_itype.add_entry(itypename.str() + " sort indices", indices);
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(),
                [&part_offsets, itype](const size_t a, const size_t b) {
                  return part_offsets[itype][a] < part_offsets[itype][b];
                });
      // now we can determine the new offsets by simply accumulating the
      // remaining sizes in the right order
      int64_t new_offset = 0;
      for (size_t i = 0; i < new_sizes[itype].size(); ++i) {
        if (nfile[0] == 1 || files[itype][indices[i]] == tfile[0]) {
          my_assert(new_offsets[itype][indices[i]] == 0,
                    "Overwriting cell offset!");
          new_offsets[itype][indices[i]] = new_offset;
          new_offset += new_sizes[itype][indices[i]];
        }
      }
    }

    // close the original snapshot, we are done with it
    CloseFile(partfile);

    // now update the remaining parts of the new snapshot
    // the number of particles in the file has changed
    group = OpenGroup(outfile, "Header");
    ReplaceArrayAttribute(group, "NumPart_ThisFile", new_numpart_thisfile);
    CloseGroup(group);
    CloseFile(outfile);

    // the total number of particles has changed as well, but since we might
    // have more files to process (maybe even on different ranks), we cannot
    // update those numbers in the file yet. We store them for now.
    for (uint_fast8_t i = 0; i < 7; ++i) {
      new_numpart_total[i] += new_numpart_thisfile[i];
    }
    for (uint_fast8_t itype = 0; itype < 5; ++itype) {
      file_offsets[itype][local_files[ifile].first] =
          new_numpart_thisfile[typelist[itype]];
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL, "Done with file loop.");
    timelog(LOGLEVEL_GENERAL, "Adjusting cell data in new files...");
  }

  // we are done processing all files (on this rank at least)
  // now accumulate the data from other ranks (if there are other ranks)
  // adjust cell data
  for (uint_fast32_t itype = 0; itype < 5; ++itype) {
    if (files[itype].size() == 0) {
      continue;
    }
    if (MPI_size > 1) {
      if (MPI_rank == 0) {
        timelog(LOGLEVEL_GENERAL, "Communicating new particle numbers...");
      }

      // accumulate cell offsets and sizes
      // it is assumed (because SWIFT guarantees it) that every rank (file) has
      // its own set of top-level cells; the particle data for one type and one
      // top-level cell cannot be spread out over multiple files
      std::vector<int64_t> global_offsets(num_cell);
      std::vector<int64_t> global_sizes(num_cell);
      int status =
          MPI_Allreduce(&new_offsets[itype][0], &global_offsets[0], num_cell,
                        MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (status != MPI_SUCCESS) {
        my_error("Error communicating new offsets!");
      }
      status = MPI_Allreduce(&new_sizes[itype][0], &global_sizes[0], num_cell,
                             MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (status != MPI_SUCCESS) {
        my_error("Error communicating new sizes!");
      }
      for (int_fast32_t icell = 0; icell < num_cell; ++icell) {
        // make sure the assumption that each cell is only present on one rank
        // is satisfied
        if (new_offsets[itype][icell] > 0 &&
            global_offsets[icell] != new_offsets[itype][icell]) {
          my_error("Multiple ranks set offset for cell!");
        }
        if (new_sizes[itype][icell] > 0 &&
            global_sizes[icell] != new_sizes[itype][icell]) {
          my_error("Multiple ranks set size for cell!");
        }
        new_offsets[itype][icell] = global_offsets[icell];
        new_sizes[itype][icell] = global_sizes[icell];
      }
      // also communicate the new number of particles in each file, which we
      // need to update the cell offsets in the virtual meta-file
      std::vector<int64_t> global_file_offsets(nfile[0]);
      status = MPI_Allreduce(&file_offsets[itype][0], &global_file_offsets[0],
                             nfile[0], MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (status != MPI_SUCCESS) {
        my_error("Error communicating new offsets!");
      }
      for (int32_t ifile = 0; ifile < nfile[0]; ++ifile) {
        if (file_offsets[itype][ifile] > 0 &&
            global_file_offsets[ifile] != file_offsets[itype][ifile]) {
          my_error("Multiple ranks set offset for file (%" PRIuFAST32
                   ", %i, %li, %li)!",
                   itype, ifile, global_file_offsets[ifile],
                   file_offsets[itype][ifile]);
        }
        file_offsets[itype][ifile] = global_file_offsets[ifile];
      }

      if (MPI_rank == 0) {
        timelog(LOGLEVEL_GENERAL, "Done with communication.");
      }
    }
  }

  // also update the new total number of particles
  uint32_t new_numpart_total_lowword[7];
  uint32_t new_numpart_total_highword[7];
  if (MPI_size > 1) {
    int64_t global_numpart[7];
    MPI_Allreduce(new_numpart_total, global_numpart, 7, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    for (uint_fast8_t i = 0; i < 7; ++i) {
      new_numpart_total[i] = global_numpart[i];
      new_numpart_total_lowword[i] = global_numpart[i];
      new_numpart_total_highword[i] = global_numpart[i] >> 32;
    }
  } else {
    for (uint_fast8_t i = 0; i < 7; ++i) {
      new_numpart_total_lowword[i] = new_numpart_total[i];
      new_numpart_total_highword[i] = new_numpart_total[i] >> 32;
    }
  }

  if (MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL,
            "Updating particle numbers in new snapshot files...");
  }
  // update the total particle number and cell meta-data for all files handled
  // by this rank
  for (size_t ifile = 0; ifile < local_files.size(); ++ifile) {
    const std::string output_file_name = compose_filename(
        output_file_prefix, ".hdf5", local_files[ifile].first, nfile[0] > 1);
    timelog(LOGLEVEL_GENERAL, "Updating file %s", output_file_name.c_str());
    const HDF5FileOrGroup outfile =
        OpenFile(output_file_name, HDF5FileModeAppend);
    for (uint_fast32_t itype = 0; itype < 5; ++itype) {
      if (new_numpart_total[typelist[itype]] == 0) {
        continue;
      }
      std::ostringstream groupnamestr;
      groupnamestr << "PartType" << typelist[itype];
      group = OpenGroup(outfile, "Cells");
      subgroup = OpenGroup(group, "Counts");
      ReplaceDataset(subgroup, groupnamestr.str(), new_sizes[itype]);
      CloseGroup(subgroup);
      subgroup = OpenGroup(group, "OffsetsInFile");
      ReplaceDataset(subgroup, groupnamestr.str(), new_offsets[itype]);
      CloseGroup(subgroup);
      CloseGroup(group);
    }
    group = OpenGroup(outfile, "Header");
    ReplaceArrayAttribute(group, "NumPart_Total", new_numpart_total_lowword);
    ReplaceArrayAttribute(group, "NumPart_Total_HighWord",
                          new_numpart_total_highword);
    CloseGroup(group);
    CloseFile(outfile);
  }
  if (MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL, "Done updating numbers.");
  }

  // now copy the virtual meta-file
  // again, we assume that everything in the file needs to be copied, except
  // for the things that we know will change
  if (nfile[0] > 1 && MPI_rank == 0) {
    timelog(LOGLEVEL_GENERAL, "Copying virtual file...");

    // convert file sizes to cumulative offsets
    for (uint_fast8_t itype = 0; itype < 5; ++itype) {
      for (int32_t ifile = 1; ifile < nfile[0]; ++ifile) {
        file_offsets[itype][ifile] += file_offsets[itype][ifile - 1];
      }
      my_assert(file_offsets[itype].back() ==
                    new_numpart_total[typelist[itype]],
                "Offsets don't match!");
    }

    const std::string old_virtual_file =
        find_file(snapshot_file_prefix, "", ".hdf5", 0, /* only_one = */ false,
                  /* use_index = */ false);
    const std::string new_virtual_file = compose_filename(
        output_file_prefix, ".hdf5", 0, /* use_index = */ false);

    const HDF5FileOrGroup old_file =
        OpenFile(old_virtual_file, HDF5FileModeRead);
    const HDF5FileOrGroup new_file =
        OpenFile(new_virtual_file, HDF5FileModeWrite);
    std::vector<std::string> blacklist;
    blacklist.push_back("PartType0");
    blacklist.push_back("PartType1");
    blacklist.push_back("PartType4");
    blacklist.push_back("PartType5");
    blacklist.push_back("PartType6");
    // general copy of the old file without the particle groups
    CopyEverythingExcept(old_file, new_file, blacklist);

    // per type copy (creation) of the new virtual particle datasets
    for (uint_fast32_t itype = 0; itype < 5; ++itype) {
      if (new_numpart_total[typelist[itype]] == 0) {
        continue;
      }
      std::ostringstream groupnamestr;
      groupnamestr << "PartType" << typelist[itype];
      timelog(LOGLEVEL_GENERAL, "%s", groupnamestr.str().c_str());
      // first copy the particle group and adjust its attributes
      CopyGroupWithoutData(old_file, new_file, groupnamestr.str());
      const HDF5FileOrGroup new_group = OpenGroup(new_file, groupnamestr.str());
      const HDF5FileOrGroup old_group = OpenGroup(old_file, groupnamestr.str());
      ReplaceArrayAttribute(new_group, "NumberOfParticles",
                            &new_numpart_total[typelist[itype]]);
      int32_t num_fields[1];
      if (AttributeExists(new_group, "NumberOfFields")) {
        ReadArrayAttribute(new_group, "NumberOfFields", num_fields);
        ++num_fields[0];
        ReplaceArrayAttribute(new_group, "NumberOfFields", num_fields);
      }
      // add the new virtual halo ID dataset and its attributes
      WriteVirtualDataset<int64_t>(new_group, "SOIDs", groupnamestr.str(),
                                   file_offsets[itype], all_files);
      HDF5FileOrGroup ds = OpenDataset(new_group, "SOIDs");
      const float float_zero = 0.0f;
      const double double_zero = 0.;
      AddAttribute(ds, "U_M exponent", float_zero);
      AddAttribute(ds, "U_L exponent", float_zero);
      AddAttribute(ds, "U_t exponent", float_zero);
      AddAttribute(ds, "U_I exponent", float_zero);
      AddAttribute(ds, "U_T exponent", float_zero);
      AddAttribute(ds, "h-scale exponent", float_zero);
      AddAttribute(ds, "a-scale exponent", float_zero);
      AddAttribute(ds, "Expression for physical CGS units", "[ - ] ");
      AddAttribute(ds, "Lossy compression filter", "None");
      AddAttribute(
          ds,
          "Conversion factor to CGS (not including cosmological corrections)",
          double_zero);
      AddAttribute(ds,
                   "Conversion factor to physical CGS (including cosmological "
                   "corrections)",
                   double_zero);
      AddAttribute(ds, "Description",
                   "ID of the most massive spherical overdensity the particles "
                   "belong to");
      CloseDataset(ds);
      std::vector<std::string> membernames;
      std::vector<std::string> memberdescr;
      membernames.push_back("GroupNr_all");
      memberdescr.push_back("Index of halo in which this particle is a member "
                            "(bound or unbound), or -1 if none");
      membernames.push_back("GroupNr_bound");
      memberdescr.push_back("Index of halo in which this particle is a bound "
                            "member, or -1 if none");
      membernames.push_back("Rank_bound");
      memberdescr.push_back("Ranking by binding energy of the bound particles "
                            "(first in halo=0), or -1 if not bound");
      for (uint_fast8_t imember = 0; imember < 3; ++imember) {
        WriteVirtualDataset<int32_t>(new_group, membernames[imember],
                                     groupnamestr.str(), file_offsets[itype],
                                     all_files);
        HDF5FileOrGroup memberds = OpenDataset(new_group, membernames[imember]);
        AddAttribute(memberds, "U_M exponent", float_zero);
        AddAttribute(memberds, "U_L exponent", float_zero);
        AddAttribute(memberds, "U_t exponent", float_zero);
        AddAttribute(memberds, "U_I exponent", float_zero);
        AddAttribute(memberds, "U_T exponent", float_zero);
        AddAttribute(memberds, "h-scale exponent", float_zero);
        AddAttribute(memberds, "a-scale exponent", float_zero);
        AddAttribute(memberds, "Expression for physical CGS units", "[ - ] ");
        AddAttribute(memberds, "Lossy compression filter", "None");
        AddAttribute(
            memberds,
            "Conversion factor to CGS (not including cosmological corrections)",
            double_zero);
        AddAttribute(
            memberds,
            "Conversion factor to physical CGS (including cosmological "
            "corrections)",
            double_zero);
        AddAttribute(memberds, "Description", memberdescr[imember].c_str());
        CloseDataset(memberds);
      }
      // copy the other particle datasets, adjusting the virtual links according
      // to the new offsets
      // again, this copy operation is agnostic of the specific datasets that
      // are present
      CopyVirtualDatasets(old_group, new_group, groupnamestr.str(),
                          file_offsets[itype], all_files);
      CloseGroup(new_group);
      CloseGroup(old_group);
    }

    CloseFile(old_file);

    // update the cell meta-data in the virtual file
    // these meta-data are the same as in the distributed files, except that
    // they now are based on a single file
    // in practice, this means we need to add an additional offset to the cell
    // offsets that depends on the file index for the particular cell
    for (uint_fast32_t itype = 0; itype < 5; ++itype) {
      if (new_numpart_total[typelist[itype]] == 0) {
        continue;
      }

      // adjust cell offsets
      for (int_fast32_t icell = 0; icell < num_cell; ++icell) {
        const int64_t file_offset =
            (files[itype][icell] > 0)
                ? file_offsets[itype][files[itype][icell] - 1]
                : 0;
        new_offsets[itype][icell] += file_offset;
      }

      std::ostringstream groupnamestr;
      groupnamestr << "PartType" << typelist[itype];
      group = OpenGroup(new_file, "Cells");
      subgroup = OpenGroup(group, "Counts");
      ReplaceDataset(subgroup, groupnamestr.str(), new_sizes[itype]);
      CloseGroup(subgroup);
      subgroup = OpenGroup(group, "OffsetsInFile");
      ReplaceDataset(subgroup, groupnamestr.str(), new_offsets[itype]);
      CloseGroup(subgroup);
      CloseGroup(group);
    }

    // finally, update the total number of particles
    // this is the same as for the distributed files
    group = OpenGroup(new_file, "Header");
    ReplaceArrayAttribute(group, "NumPart_Total", new_numpart_total_lowword);
    ReplaceArrayAttribute(group, "NumPart_Total_HighWord",
                          new_numpart_total_highword);
    ReplaceArrayAttribute(group, "NumPart_ThisFile", new_numpart_total);
    CloseGroup(group);
    CloseFile(new_file);
  }

  // we are all done (on this rank)
  timelog(LOGLEVEL_GENERAL, "Done.");

#ifdef OUTPUT_MEMORY_LOG
  // output the memory log (for this rank)
  std::stringstream memory_name;
  memory_name << "memory_log.";
  memory_name.width(std::ceil(std::log10(MPI_size)));
  memory_name.fill('0');
  memory_name << MPI_rank;
  memory_file.dump(memory_name.str());
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  if (MPI_rank == 0) {
    std::cout << "Done." << std::endl;
    std::cout << "Took " << global_timer.current_time() << " ms." << std::endl;
  }

  // wait for the other ranks and exit
  return MPI_Finalize();
}
