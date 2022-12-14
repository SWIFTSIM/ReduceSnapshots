/**
 * @file HDF5.hpp
 *
 * @brief Custom HDF5 wrapper.
 *
 * We encapsulate most HDF5 calls in more intuitive functions and hide all the
 * error handling. And we use templates to facilitate some of the data type
 * handling.
 *
 * @author Bert Vandenbroucke (vandenbroucke@strw.leidenuniv.nl)
 */

#ifndef RS_HDF5_HPP
#define RS_HDF5_HPP

#include "Error.hpp"

#include <algorithm>
#include <cstring>
#include <hdf5.h>
#include <sstream>
#include <string>
#include <vector>

/*! @brief In principle, using the HDF5 core driver for file operations should
 *  make read and write operations faster, since the entire file is stored in
 *  memory. In practice, using the core driver for large files seems to result
 *  in weird errors, so it is better to not use this. */
//#define HDF5_USE_CORE_DRIVER

namespace HDF5Datatypes {

/**
 * @brief Get the HDF5 data type corresponding to the template typename.
 *
 * This template function needs to be specialized for every typename that is
 * used.
 *
 * @return hid_t handle for the corresponding HDF5 data type.
 */
template <typename _datatype_> inline hid_t get_datatype_name();

/**
 * @brief get_datatype_name specialization for a double precision floating point
 * value.
 *
 * @return H5T_NATIVE_DOUBLE.
 */
template <> inline hid_t get_datatype_name<double>() {
  return H5T_NATIVE_DOUBLE;
}

/**
 * @brief get_datatype_name specialization for a single precision floating point
 * value.
 *
 * @return H5T_NATIVE_FLOAT.
 */
template <> inline hid_t get_datatype_name<float>() { return H5T_NATIVE_FLOAT; }

/**
 * @brief get_datatype_name specialization for a 32 bit unsigned integer.
 *
 * @return H5T_NATIVE_UINT32.
 */
template <> inline hid_t get_datatype_name<uint32_t>() {
  return H5T_NATIVE_UINT32;
}

/**
 * @brief get_datatype_name specialization for a 64 bit unsigned integer.
 *
 * @return H5T_NATIVE_UINT64.
 */
template <> inline hid_t get_datatype_name<uint64_t>() {
  return H5T_NATIVE_UINT64;
}

/**
 * @brief get_datatype_name specialization for a 32 bit signed integer.
 *
 * @return H5T_NATIVE_INT32.
 */
template <> inline hid_t get_datatype_name<int32_t>() {
  return H5T_NATIVE_INT32;
}

/**
 * @brief get_datatype_name specialization for a 64 bit signed integer.
 *
 * @return H5T_NATIVE_INT64.
 */
template <> inline hid_t get_datatype_name<int64_t>() {
  return H5T_NATIVE_INT64;
}
} // namespace HDF5Datatypes

namespace HDF5 {

/*! @brief More convenient name for a HDF5 file or group handle. */
typedef hid_t HDF5FileOrGroup;

/*! @brief More convenient name for a HDF5 dataset handle. */
typedef hid_t HDF5Dataset;

/**
 * @brief Possible HDF5 file open modes.
 */
enum HDF5FileMode {
  /*! @brief Open the file for reading only. */
  HDF5FileModeRead,
  /*! @brief Open the file for writing, overwriting existing files. */
  HDF5FileModeWrite,
  /*! @brief Open the file for appending: the file is assumed to already exist,
   *  but is opened with write permissions. */
  HDF5FileModeAppend
};

/**
 * @brief Initialise our HDF5 wrapper.
 *
 * Should be called before any other HDF5 wrapper function is used.
 *
 * No harm is done by not calling this function, it simply overwrites the
 * default error handler, so that we can display more useful HDF5 error
 * messages.
 */
inline void StartHDF5() {
  const herr_t hdf5status = H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
  if (hdf5status < 0) {
    my_error("Unable to turn off default HDF5 error handling!");
  }
}

/**
 * @brief Open an HDF5 file.
 *
 * The file should be closed using CloseFile() when it is no longer needed.
 *
 * @param filename Name of the file.
 * @param mode Mode in which the file is opened.
 * @return File handle that should be used for future operations that involve
 * the file.
 */
inline HDF5FileOrGroup OpenFile(const std::string filename,
                                const HDF5FileMode mode) {
  hid_t file;
#ifdef HDF5_USE_CORE_DRIVER
  // open the file using the HDF5 core driver, with a default memory incrememt
  // size of 500k bytes
  const hid_t props = H5Pcreate(H5P_FILE_ACCESS);
  herr_t hdf5status = H5Pset_fapl_core(props, 500000, 1);
#else
  // use default file open properties
  const hid_t props = H5P_DEFAULT;
#endif
  switch (mode) {
  case HDF5FileModeRead:
    // open the file for reading
    file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, props);
    break;
  case HDF5FileModeWrite:
    // create a new file, overwriting existing onces
    file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, props);
    break;
  case HDF5FileModeAppend:
    // open a file with read-write access
    file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, props);
    break;
  default:
    my_error("Unknown file mode!");
    file = 0;
    break;
  }
  if (file < 0) {
    my_error("Unable to open file \"%s\"!", filename.c_str());
  }
#ifdef HDF5_USE_CORE_DRIVER
  hdf5status = H5Pclose(props);
  if (hdf5status < 0) {
    my_error("Error closing file access properties!");
  }
#endif
  return file;
}

/**
 * @brief Close an HDF5 file.
 *
 * After calling this function, subsequent operations that use the file handle
 * will result in errors.
 *
 * @param file File handle.
 */
inline void CloseFile(const HDF5FileOrGroup file) {
  const herr_t hdf5status = H5Fclose(file);
  if (hdf5status < 0) {
    my_error("Unable to close file!");
  }
}

/**
 * @brief Open a group within an HDF5 file.
 *
 * The group should be closed using CloseGroup() when it is no longer needed.
 *
 * @param file File handle.
 * @param name Name of the group.
 * @return Group handle that should be used for all future operations that
 * involve this group.
 */
inline HDF5FileOrGroup OpenGroup(const HDF5FileOrGroup file,
                                 const std::string name) {
  const hid_t group = H5Gopen(file, name.c_str(), H5P_DEFAULT);
  if (group < 0) {
    my_error("Error opening \"%s\" group!", name.c_str());
  }
  return group;
}

/**
 * @brief Create a group within an HDF5 file.
 *
 * The group should be closed using CloseGroup() when it is no longer needed.
 *
 * @param file File handle.
 * @param name Name of the group.
 * @return Group handle that should be used for all future operations that
 * involve this group.
 */
inline HDF5FileOrGroup CreateGroup(const HDF5FileOrGroup file,
                                   const std::string name) {
  const hid_t group =
      H5Gcreate(file, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (group < 0) {
    my_error("Error creating \"%s\" group!", name.c_str());
  }
  return group;
}

/**
 * @brief Close an HDF5 group.
 *
 * After this function has been called, subsequent operations involving the
 * group handle will result in errors.
 *
 * @param group Group handle.
 */
inline void CloseGroup(const HDF5FileOrGroup group) {
  const herr_t hdf5status = H5Gclose(group);
  if (hdf5status < 0) {
    my_error("Error closing header group!");
  }
}

inline uint64_t GetDatasetSize(const HDF5FileOrGroup file,
                               const std::string name) {
  const hid_t dset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    my_error("Unable to open dataset \"%s\"!", name.c_str());
  }
  const hid_t space = H5Dget_space(dset);
  if (space < 0) {
    my_error("Unable to obtain data space for dataset \"%s\"!", name.c_str());
  }
  hsize_t dims[2];
  const int ndim = H5Sget_simple_extent_dims(space, dims, nullptr);
  if (ndim != 1) {
    my_error("Expected dataset \"%s\" to be 1D!", name.c_str());
  }
  herr_t hdf5status = H5Sclose(space);
  if (hdf5status < 0) {
    my_error("Error closing data space for dataset \"%s\"!", name.c_str());
  }
  hdf5status = H5Dclose(dset);
  if (hdf5status < 0) {
    my_error("Error closing dataset \"%s\"!", name.c_str());
  }
  return dims[0];
}

/**
 * @brief Read a single value from an HDF5 dataset.
 *
 * @tparam _type_ Type of the value that is to be read.
 * @param file File or group handle that contains the dataset.
 * @param name Name of the dataset.
 * @return Data value that was read.
 */
template <typename _type_>
inline _type_ ReadSingleValueDataset(const HDF5FileOrGroup file,
                                     const std::string name) {

  // get the appropriate HDF5 type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();

  // open the dataset
  const hid_t dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Unable to open dataset \"%s\"!", name.c_str());
  }
  // read the (single) value
  // we use the () constructor to deal with non-standard types (which will
  // likely cause other problems, but hey)
  _type_ value(0);
  herr_t hdf5status =
      H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
  if (hdf5status < 0) {
    my_error("Error reading dataset \"%s\"!", name.c_str());
  }
  hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Error closing dataset!");
  }
  return value;
}

/**
 * @brief Read an entire HDF5 dataset into a 1D vector.
 *
 * Note that this function works regardless of the dimensions of the dataset.
 * The dataset will be flattened into a 1D vector in standard C order, i.e. for
 * a 2D dataset, the output vector will contain all columns of the first row,
 * followed by all columns of the second row and so on. For example, if you
 * read the Coordinates dataset, the resulting vector will contain
 *  x1, y1, z1, x2, y2, z2...
 *
 * @tparam _type_ Type of the data values.
 * @param file File or group handle that contains the dataset.
 * @param name Name of the dataset.
 * @param value Pre-allocated vector of at least the right size to hold the
 * data.
 */
template <typename _type_>
inline void ReadEntireDataset(const HDF5FileOrGroup file,
                              const std::string name,
                              std::vector<_type_> &value) {

  // get the HDF5 data type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();

  // open the dataset
  const hid_t dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Unable to open dataset \"%s\"!", name.c_str());
  }
  // read the data
  // we assume the given vector is large enough to hold the data
  herr_t hdf5status =
      H5Dread(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value[0]);
  if (hdf5status < 0) {
    my_error("Error reading dataset \"%s\"!", name.c_str());
  }
  hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Error closing dataset!");
  }
}

/**
 * @brief Read part of an HDF5 dataset into a 1D vector.
 *
 * @tparam _type_ Type of the data values.
 * @param file File or group handle that contains the dataset.
 * @param name Name of the dataset.
 * @param offset Start of the chunk we want to read, as a number of elements
 * from the start of the data array.
 * @param size Number of elements to read.
 * @param value Pre-allocated vector of at least the right size to hold the
 * result, i.e. size.
 */
template <typename _type_>
inline void ReadPartialDataset(const HDF5FileOrGroup file,
                               const std::string name, const size_t offset,
                               const size_t size, std::vector<_type_> &value) {

  // get the HDF5 data type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();

  // open the dataset
  const hid_t dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Unable to open dataset!");
  }
  // select a hyperslab within the dataset starting from the given offset and
  // with the given size
  const hid_t filespace = H5Dget_space(dataset);
  if (filespace < 0) {
    my_error("Could not access file space!");
  }
  const hsize_t dims[1] = {size};
  const hsize_t offs[1] = {offset};
  herr_t hdf5status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offs,
                                          nullptr, dims, nullptr);
  if (hdf5status < 0) {
    my_error("Failed to select hyperslab!");
  }
  // create a memory space of the right dimensions to process the chunk
  const hid_t memspace = H5Screate_simple(1, dims, nullptr);
  if (memspace < 0) {
    my_error("Failed to allocate memory space!");
  }
  // now read the dataset
  hdf5status =
      H5Dread(dataset, dtype, memspace, filespace, H5P_DEFAULT, &value[0]);
  if (hdf5status < 0) {
    my_error("Error reading partial dataset \"%s\"!", name.c_str());
  }
  hdf5status = H5Sclose(memspace);
  if (hdf5status < 0) {
    my_error("Failed to close memory space!");
  }
  hdf5status = H5Sclose(filespace);
  if (hdf5status < 0) {
    my_error("Failed to close file space!");
  }
  hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Error closing dataset!");
  }
}

/**
 * @brief Read part of a 3D HDF5 dataset into a 1D vector.
 *
 * 3D in this case means a 2D dataset with 3 columns per row, i.e. of shape
 * (N, 3). Examples are the Coordinates and Velocities datasets in a SWIFT
 * snapshot.
 *
 * The resulting array will contain the data values in standard C order, i.e.
 * first all columns for the first row, then all columns for the second row and
 * so on. For the coordinates for example, this means
 *  x1, y1, z1, x2, y2, z2...
 *
 * @tparam _type_ Type of the data values.
 * @param file File or group handle that contains the dataset.
 * @param name Name of the dataset.
 * @param offset Start of the chunk we want to read, as a number of rows from
 * the start of the data array.
 * @param size Number of rows to read.
 * @param value Pre-allocated vector of at least the right size to hold the
 * result, i.e. 3 * size.
 */
template <typename _type_>
inline void ReadPartial3DDataset(const HDF5FileOrGroup file,
                                 const std::string name, const size_t offset,
                                 const size_t size,
                                 std::vector<_type_> &value) {

  // get the HDF5 data type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();

  // open the dataset
  const hid_t dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Unable to open dataset!");
  }
  // select the hyperslab within the dataset that we want to read
  const hid_t filespace = H5Dget_space(dataset);
  if (filespace < 0) {
    my_error("Could not access file space!");
  }
  const hsize_t dims[2] = {size, 3};
  const hsize_t offs[2] = {offset, 0};
  herr_t hdf5status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offs,
                                          nullptr, dims, nullptr);
  if (hdf5status < 0) {
    my_error("Failed to select hyperslab!");
  }
  // create a memory space of the same dimensions to process the data
  const hid_t memspace = H5Screate_simple(2, dims, nullptr);
  if (memspace < 0) {
    my_error("Failed to allocate memory space!");
  }
  // now read the data
  hdf5status =
      H5Dread(dataset, dtype, memspace, filespace, H5P_DEFAULT, &value[0]);
  if (hdf5status < 0) {
    my_error("Error reading partial dataset \"%s\"!", name.c_str());
  }
  hdf5status = H5Sclose(memspace);
  if (hdf5status < 0) {
    my_error("Failed to close memory space!");
  }
  hdf5status = H5Sclose(filespace);
  if (hdf5status < 0) {
    my_error("Failed to close file space!");
  }
  hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Error closing dataset!");
  }
}

/**
 * @brief Auxiliary struct to facilitate working with chunks in an HDF5
 * dataset.
 */
struct HDF5Chunk {
  /*! @brief Offset of the chunk. */
  size_t offset;
  /*! @brief Size of the chunk. */
  size_t size;
};

/**
 * @brief Read a partial 1D HDF5 dataset based on a list of chunks.
 *
 * Note that while the chunks can be passed on in arbitrary order, the HDF5
 * library will automatically sort the resulting values based on order in the
 * file. This makes sense if you think about the chunks as being selection
 * regions: you can tell HDF5 to select parts of the dataset in an arbitrary
 * order, but the entire selection will still have a pre-defined order and
 * that order determines the order of the values that are read.
 *
 * Since it would be too inefficient to sort the data after reading, the
 * caller needs to deal with a possible mismatch between chunk and data order.
 *
 * @tparam _type_ Type of the data values.
 * @param file File or group handle that contains the dataset.
 * @param name Name of the dataset.
 * @param chunks List of chunks to read (in arbitrary order).
 * @param value Pre-allocated vector of at least the right size to hold the
 * result, i.e. the sum of all chunk sizes.
 */
template <typename _type_>
inline void ReadPartialDataset(const HDF5FileOrGroup file,
                               const std::string name,
                               const std::vector<struct HDF5Chunk> &chunks,
                               std::vector<_type_> &value) {

  // get the HDF5 data type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();

  // open the dataset
  const hid_t dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Unable to open dataset!");
  }
  // open the data space, i.e. the HDF5 abstraction of the data array
  const hid_t filespace = H5Dget_space(dataset);
  if (filespace < 0) {
    my_error("Could not access file space!");
  }
  // now select hyperslabs within the data space corresponding to the chunks
  // we want to read
  // we start by selecting the first chunk (which should always exist)
  // we then add the other chunks to the selection using a logical or
  hsize_t dims[1] = {chunks[0].size};
  hsize_t offs[1] = {chunks[0].offset};
  // we need to determine the total size that will be read
  hsize_t total_size = chunks[0].size;
  herr_t hdf5status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offs,
                                          nullptr, dims, nullptr);
  if (hdf5status < 0) {
    my_error("Failed to select hyperslab!");
  }
  for (size_t i = 1; i < chunks.size(); ++i) {
    dims[0] = chunks[i].size;
    offs[0] = chunks[i].offset;
    total_size += chunks[i].size;
    hdf5status = H5Sselect_hyperslab(filespace, H5S_SELECT_OR, offs, nullptr,
                                     dims, nullptr);
    if (hdf5status < 0) {
      my_error("Failed to select hyperslab!");
    }
  }
  my_assert(total_size == value.size(), "Size mismatch!");
  dims[0] = total_size;
  // create a memory space that is large enough to read the data
  const hid_t memspace = H5Screate_simple(1, dims, nullptr);
  if (memspace < 0) {
    my_error("Failed to allocate memory space!");
  }
  // now read the data
  hdf5status =
      H5Dread(dataset, dtype, memspace, filespace, H5P_DEFAULT, &value[0]);
  if (hdf5status < 0) {
    my_error("Error reading partial dataset \"%s\"!", name.c_str());
  }
  hdf5status = H5Sclose(memspace);
  if (hdf5status < 0) {
    my_error("Failed to close memory space!");
  }
  hdf5status = H5Sclose(filespace);
  if (hdf5status < 0) {
    my_error("Failed to close file space!");
  }
  hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Error closing dataset!");
  }
}

/**
 * @brief Read a partial 3D HDF5 dataset based on a list of chunks.
 *
 * 3D in this case means a 2D dataset with 3 columns per row, i.e. of shape
 * (N, 3). Examples are the Coordinates and Velocities datasets in a SWIFT
 * snapshot.
 *
 * The resulting array will contain the data values in standard C order, i.e.
 * first all columns for the first row, then all columns for the second row and
 * so on. For the coordinates for example, this means
 *  x1, y1, z1, x2, y2, z2...
 *
 * Note that while the chunks can be passed on in arbitrary order, the HDF5
 * library will automatically sort the resulting values based on order in the
 * file. This makes sense if you think about the chunks as being selection
 * regions: you can tell HDF5 to select parts of the dataset in an arbitrary
 * order, but the entire selection will still have a pre-defined order and
 * that order determines the order of the values that are read.
 *
 * Since it would be too inefficient to sort the data after reading, the
 * caller needs to deal with a possible mismatch between chunk and data order.
 *
 * @tparam _type_ Type of the data values.
 * @param file File or group handle that contains the dataset.
 * @param name Name of the dataset.
 * @param chunks List of chunks to read (in arbitrary order).
 * @param value Pre-allocated vector of at least the right size to hold the
 * result, i.e. the sum of all chunk sizes.
 */
template <typename _type_>
inline void ReadPartial3DDataset(const HDF5FileOrGroup file,
                                 const std::string name,
                                 const std::vector<struct HDF5Chunk> &chunks,
                                 std::vector<_type_> &value) {

  // get the HDF5 data type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();

  // open the dataset
  const hid_t dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Unable to open dataset!");
  }
  // open the data space
  const hid_t filespace = H5Dget_space(dataset);
  if (filespace < 0) {
    my_error("Could not access file space!");
  }
  // now select the chunks as hyperslabs, starting from the first chunk (which
  // is assumed to always exist)
  hsize_t dims[2] = {chunks[0].size, 3};
  hsize_t offs[2] = {chunks[0].offset, 0};
  // accumulate the total size of all chunks
  hsize_t total_size = chunks[0].size;
  herr_t hdf5status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offs,
                                          nullptr, dims, nullptr);
  if (hdf5status < 0) {
    my_error("Failed to select hyperslab!");
  }
  // add additional chunks using logical or
  for (size_t i = 1; i < chunks.size(); ++i) {
    dims[0] = chunks[i].size;
    offs[0] = chunks[i].offset;
    total_size += chunks[i].size;
    hdf5status = H5Sselect_hyperslab(filespace, H5S_SELECT_OR, offs, nullptr,
                                     dims, nullptr);
    if (hdf5status < 0) {
      my_error("Failed to select hyperslab!");
    }
  }
  my_assert(3 * total_size == value.size(), "Size mismatch!");
  dims[0] = total_size;
  // create a memory space large enough to deal with all chunks
  const hid_t memspace = H5Screate_simple(2, dims, nullptr);
  if (memspace < 0) {
    my_error("Failed to allocate memory space!");
  }
  // now read the data
  hdf5status =
      H5Dread(dataset, dtype, memspace, filespace, H5P_DEFAULT, &value[0]);
  if (hdf5status < 0) {
    my_error("Error reading partial dataset \"%s\"!", name.c_str());
  }
  hdf5status = H5Sclose(memspace);
  if (hdf5status < 0) {
    my_error("Failed to close memory space!");
  }
  hdf5status = H5Sclose(filespace);
  if (hdf5status < 0) {
    my_error("Failed to close file space!");
  }
  hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Error closing dataset!");
  }
}

/**
 * @brief Read an HDF5 attribute.
 *
 * @tparam _type_ Attribute data type.
 * @param group Group or dataset handle that contains the attribute.
 * @param name Name of the attribute.
 * @param value Pointer to an array that is large enough to store the attribute
 * value(s).
 */
template <typename _type_>
inline void ReadArrayAttribute(const HDF5FileOrGroup group,
                               const std::string name, _type_ *value) {

  // determine the HDF5 data type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();
  // open the attribute
  const hid_t attr = H5Aopen(group, name.c_str(), H5P_DEFAULT);
  if (attr < 0) {
    my_error("Error opening attribute \"%s\"!", name.c_str());
  }
  // read the attribute
  herr_t hdf5status = H5Aread(attr, dtype, value);
  if (hdf5status < 0) {
    my_error("Error reading attribute \"%s\"!", name.c_str());
  }
  hdf5status = H5Aclose(attr);
  if (hdf5status < 0) {
    my_error("Error closing attribute!");
  }
}

/**
 * @brief Replace an HDF5 attribute with new values.
 *
 * @tparam _type_ Attribute data type.
 * @param group Group or dataset handle that contains the attribute.
 * @param name Name of the attribute.
 * @param value Pointer to an array with new values, at least large enough to
 * have values for all elements in the attribute array.
 */
template <typename _type_>
inline void ReplaceArrayAttribute(const HDF5FileOrGroup group,
                                  const std::string name, const _type_ *value) {

  // determine the HDF5 type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();
  // open the attribute
  const hid_t attr = H5Aopen(group, name.c_str(), H5P_DEFAULT);
  if (attr < 0) {
    my_error("Error opening attribute \"%s\"!", name.c_str());
  }
  // write new attribute values
  herr_t hdf5status = H5Awrite(attr, dtype, value);
  if (hdf5status < 0) {
    my_error("Error writing attribute \"%s\"!", name.c_str());
  }
  hdf5status = H5Aclose(attr);
  if (hdf5status < 0) {
    my_error("Error closing attribute!");
  }
}

/**
 * @brief Auxiliary class used in CopyEverythingExcept() and HDF5CopyLink().
 *
 * This class holds information that cannot be passed on to HDF5CopyLink()
 * directly and is instead passed on using a standard C void pointer
 * technique.
 */
class HDF5CopyData {
private:
  /*! @brief File handle of the destination file to which data is being
   *  copied. */
  const HDF5FileOrGroup _output_file;
  /*! @brief List of objects that should not be copied over. */
  const std::vector<std::string> &_blacklist;

public:
  /**
   * @brief Constructor.
   *
   * @param output_file Destination file handle.
   * @param blacklist List of objects that should not be copied over.
   */
  inline HDF5CopyData(const HDF5FileOrGroup output_file,
                      const std::vector<std::string> &blacklist)
      : _output_file(output_file), _blacklist(blacklist) {}

  /**
   * @brief Get the destination file handle.
   *
   * @return Destination file handle.
   */
  inline HDF5FileOrGroup get_output_file() const { return _output_file; }

  /**
   * @brief Check if the given object should be copied over.
   *
   * @param link_name Name of an object in the HDF5 file.
   * @return True if the object should be copied over.
   */
  inline bool is_blacklisted(const std::string link_name) const {
    return std::count(_blacklist.begin(), _blacklist.end(), link_name) > 0;
  }
};

/**
 * @brief Recursive HDF5 copy function, called using H5Literate() from
 * CopyEverythingExcept().
 *
 * @param group_id File or group handle of the source file we are copying
 * from.
 * @param link_name Name of the object currently being looked at by
 * H5Literate().
 * @param link_info (Mostly useless) additional information about the object
 * currently being looked at.
 * @param extra_data Additional information passed on to the function, i.e. our
 * void* casted HDF5CopyData object.
 * @return H5Literate() compatible return value: 0 if the object was processed
 * correctly and we want to proceed with the next iteration, -1 if there was an
 * error and we want to abort, 1 if we want to stop the iteration early (which
 * we do not use here).
 */
inline herr_t HDF5CopyLink(hid_t group_id, const char *link_name,
                           const H5L_info_t *link_info, void *extra_data) {

  // recover the HDF5CopyData object
  HDF5CopyData *data = reinterpret_cast<HDF5CopyData *>(extra_data);
  // check if action is required for this object
  if (data->is_blacklisted(link_name)) {
    my_statusmessage("Not copying blacklisted group \"%s\".", link_name);
    return 0;
  }
  my_statusmessage("Copying \"%s\"...", link_name);
  herr_t hdf5status;
  // copy over the object depending on its type
  if (link_info->type == H5L_TYPE_SOFT) {
    my_statusmessage("Soft link: soft copy.");
    // a soft link is copied as a link
    hdf5status = H5Lcopy(group_id, link_name, data->get_output_file(),
                         link_name, H5P_DEFAULT, H5P_DEFAULT);
  } else {
    my_statusmessage("Hard link: real copy.");
    // a hard link is a real object that needs a real copy
    hdf5status = H5Ocopy(group_id, link_name, data->get_output_file(),
                         link_name, H5P_DEFAULT, H5P_DEFAULT);
  }
  if (hdf5status < 0) {
    my_errormessage("Error copying \"%s\"!", link_name);
    return -1;
  }
  my_statusmessage("Done.");
  return 0;
}

/**
 * @brief Copy all contents of the given input file into the given output file,
 * except for objects with names that are blacklisted.
 *
 * @param input_file Input file handle.
 * @param output_file Output file handle.
 * @param blacklist List of objects that should not be copied over.
 */
inline void CopyEverythingExcept(const HDF5FileOrGroup input_file,
                                 const HDF5FileOrGroup output_file,
                                 const std::vector<std::string> &blacklist) {
  // create an HDF5CopyData object that contains additional arguments to
  // HDF5CopyLink()
  HDF5CopyData copy_data(output_file, blacklist);
  // now iterate over the file contents and copy what needs to be copied
  const herr_t hdf5status =
      H5Literate(input_file, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr,
                 HDF5CopyLink, &copy_data);
  if (hdf5status < 0) {
    my_error("Error during file copy iteration!");
  }
}

/**
 * @brief Copy all contents of the given input file into the given output file.
 *
 * (Literally) equivalent to calling CopyEverythingExcept() with an empty
 * blacklist.
 *
 * @param input_file Input file handle.
 * @param output_file Output file handle.
 */
inline void CopyEverything(const HDF5FileOrGroup input_file,
                           const HDF5FileOrGroup output_file) {
  CopyEverythingExcept(input_file, output_file, std::vector<std::string>());
}

/**
 * @brief Auxiliary object used by HDF5CopyAttribute() and CopyAttributes().
 *
 * This object acts as the additional data that is passed on to
 * HDF5CopyAttribute() by H5Aiterate() using a standard C void pointer cast.
 */
class HDF5AttributeCopyData {
private:
  /*! @brief Destination group/dataset handle we are copying to. */
  const HDF5FileOrGroup _output_group;

public:
  /**
   * @brief Constructor.
   *
   * @param output_group Destination group/dataset handle.
   */
  inline HDF5AttributeCopyData(const HDF5FileOrGroup output_group)
      : _output_group(output_group) {}

  /**
   * @brief Get the destination group/dataset handle.
   *
   * @return Destination group/dataset handle.
   */
  inline HDF5FileOrGroup get_output_group() const { return _output_group; }
};

/**
 * @brief Recursive attribute copy function, called by H5Aiterate() from
 * CopyAttributes().
 *
 * @param group_id Source group/dataset handle.
 * @param name Name of the attribute we are currently looking at.
 * @param ainfo Extra information for the current attribute.
 * @param extra_data Void* casted HDF5AttributeCopyData object.
 * @return H5Aiterate() compatible return value: 0 if the attribute was
 * processed correctly and we want to proceed with the next iteration, -1 if
 * there was an error and we want to abort, 1 if we want to stop the iteration
 * early (which we do not use here).
 */
inline herr_t HDF5CopyAttribute(hid_t group_id, const char *name,
                                const H5A_info_t *ainfo, void *extra_data) {

  my_statusmessage("Copying attribute \"%s\".", name);
  // recover the HDF5AttributeCopyData object
  HDF5AttributeCopyData *data =
      reinterpret_cast<HDF5AttributeCopyData *>(extra_data);
  // open the old attribute
  const hid_t old_attr = H5Aopen(group_id, name, H5P_DEFAULT);
  if (old_attr < 0) {
    my_errormessage("Error opening attribute \"%s\"!", name);
    return -1;
  }
  // query the old attribute type
  const hid_t atype = H5Aget_type(old_attr);
  if (atype < 0) {
    my_errormessage("Error getting attribute datatype!");
    return -1;
  }
  // open the old attribute data space
  const hid_t old_aspace = H5Aget_space(old_attr);
  if (old_aspace < 0) {
    my_errormessage("Error getting attribute dataspace!");
    return -1;
  }
  // create a new attribute data space based on the old data space
  const hid_t new_aspace = H5Scopy(old_aspace);
  if (new_aspace < 0) {
    my_errormessage("Error copying attribute dataspace!");
    return -1;
  }
  // create the new attribute
  const hid_t new_attr = H5Acreate(data->get_output_group(), name, atype,
                                   new_aspace, H5P_DEFAULT, H5P_DEFAULT);
  if (new_attr < 0) {
    my_errormessage("Error creating attribute \"%s\"!", name);
    return -1;
  }
  // read the old attribute into a type-less buffer
  std::vector<char> buffer(ainfo->data_size);
  herr_t hdf5status = H5Aread(old_attr, atype, &buffer[0]);
  if (hdf5status < 0) {
    my_errormessage("Error reading attribute \"%s\"!", name);
    return -1;
  }
  // write the buffer into the new attribute
  hdf5status = H5Awrite(new_attr, atype, &buffer[0]);
  if (hdf5status < 0) {
    my_errormessage("Error writing attribute \"%s\"!", name);
    return -1;
  }
  hdf5status = H5Sclose(new_aspace);
  if (hdf5status < 0) {
    my_errormessage("Error closing attribute dataspace!");
    return -1;
  }
  hdf5status = H5Aclose(new_attr);
  if (hdf5status < 0) {
    my_errormessage("Error closing attribute \"%s\"!", name);
    return -1;
  }
  hdf5status = H5Sclose(old_aspace);
  if (hdf5status < 0) {
    my_errormessage("Error closing attribute dataspace!");
    return -1;
  }
  hdf5status = H5Tclose(atype);
  if (hdf5status < 0) {
    my_errormessage("Error closing attribute datatype!");
    return -1;
  }
  hdf5status = H5Aclose(old_attr);
  if (hdf5status < 0) {
    my_errormessage("Error closing attribute \"%s\"!", name);
    return -1;
  }
  return 0;
}

/**
 * @brief Copy all attributes from the given input group/dataset to the given
 * output group/dataset.
 *
 * @param input_group Input group/dataset handle.
 * @param output_group Output group/dataset handle.
 */
inline void CopyAttributes(const HDF5FileOrGroup input_group,
                           const HDF5FileOrGroup output_group) {

  // Create an HDF5AttributeCopyData object with additional arguments for
  // HDF5CopyAttribute()
  HDF5AttributeCopyData copy_data(output_group);
  // now iterate over the attributes and copy them
  herr_t hdf5status = H5Aiterate(input_group, H5_INDEX_NAME, H5_ITER_NATIVE,
                                 nullptr, HDF5CopyAttribute, &copy_data);
  if (hdf5status < 0) {
    my_error("Error copying attributes!");
  }
}

/**
 * @brief Copy the group with the given name from the given input file to the
 * given output file, keeping all its attributes but not copying over any
 * datasets it contains.
 *
 * @param input_file Input file/group handle.
 * @param output_file Output file/group handle.
 * @param name Name of the group.
 */
inline void CopyGroupWithoutData(const HDF5FileOrGroup input_file,
                                 const HDF5FileOrGroup output_file,
                                 const std::string name) {

  // Create a new group in the output file
  const hid_t new_group = H5Gcreate(output_file, name.c_str(), H5P_DEFAULT,
                                    H5P_DEFAULT, H5P_DEFAULT);
  if (new_group < 0) {
    my_error("Failed to created group \"%s\"!", name.c_str());
  }
  // open the group in the old file
  const hid_t old_group = OpenGroup(input_file, name);
  // copy the attributes from the old group to the new group
  CopyAttributes(old_group, new_group);
  CloseGroup(old_group);
  const herr_t hdf5status = H5Gclose(new_group);
  if (hdf5status < 0) {
    my_error("Failed to close group \"%s\"!", name.c_str());
  }
}

/**
 * @brief Auxiliary object used by HDF5CopyResizedDataset() and CopyDatasets()
 * to copy partial datasets from one file to another.
 */
class HDF5DatasetCopyData {
private:
  /*! @brief Destination file/group handle we are copying to. */
  const HDF5FileOrGroup _output_group;
  /*! @brief Size of the new datasets, should be smaller or equal to the size of
   *  the old datasets. */
  const size_t _new_size;
  /*! @brief Mask that specifies which elements in the old dataset need to be
   *  copied over. */
  const std::vector<bool> &_mask;
  /*! @brief Maximum number of elements that can be read from the old dataset
   *  in one go. This also puts a limit on the maximum number of elements that
   *  is written to the new dataset at once, although the exact number depends
   *  on the mask for each particular chunk. */
  const size_t _max_chunksize;
  /*! @brief Memory logging variable that counts the maximum memory usage of
   *  the chunks we read and write. */
  size_t _copy_size;

public:
  /**
   * @brief Constructor.
   *
   * @param output_group Destination file/group handle.
   * @param new_size Number of elements in the new dataset, at most the same as
   * the number of elements in the old dataset.
   * @param mask Mask that specifies which elements of the old dataset are to be
   * copied over.
   * @param max_chunksize Maximum allowed size of a single chunk of the old
   * dataset that is read.
   */
  inline HDF5DatasetCopyData(const HDF5FileOrGroup output_group,
                             const size_t new_size,
                             const std::vector<bool> &mask,
                             const size_t max_chunksize)
      : _output_group(output_group), _new_size(new_size), _mask(mask),
        _max_chunksize(max_chunksize), _copy_size(0) {}

  /**
   * @brief Get the destination file/group handle.
   *
   * @return Destination file/group handle.
   */
  inline HDF5FileOrGroup get_output_group() const { return _output_group; }

  /**
   * @brief Get the size of the new datasets.
   *
   * @return Number of elements of a new dataset, i.e. number of elements that
   * needs to be copied over.
   */
  inline size_t get_new_size() const { return _new_size; }

  /**
   * @brief Keep the element with the given index?
   *
   * @param i Index of an element in the old dataset.
   * @return True if this element should be written to the new dataset.
   */
  inline size_t keep(const size_t i) const { return _mask[i]; }

  /**
   * @brief Get the maximally allowed chunk size.
   *
   * @return Maximum size of a single chunk of the old dataset that can be read
   * in one go.
   */
  inline size_t get_max_chunksize() const { return _max_chunksize; }

  /**
   * @brief Log the memory usage of a single chunk copy operation.
   *
   * @param size Size (in bytes) used by the copy operation.
   */
  inline void register_size(const size_t size) {
    _copy_size = std::max(_copy_size, size);
  }

  /**
   * @brief Get the maximum size of any copy iterations.
   *
   * @return Maximum memory usage of any of the copy iterations.
   */
  inline size_t get_copy_size() const { return _copy_size; }
};

/**
 * @brief H5Literate() compatible copy function that copies over masked chunks
 * of a dataset to a new file.
 *
 * @param group_id Source file/group handle.
 * @param link_name Name of the dataset currently being looked at.
 * @param link_info Additional information about the object currently being
 * looked at.
 * @param extra_data Void* casted HDF5DatasetCopyData object.
 * @return H5Literate() compatible return value: 0 if the object was processed
 * successfully and we want to proceed with the next iteration, -1 if an error
 * occurred and we want to abort, 1 if we want to prematurely stop iterating
 * (which we do not use here).
 */
inline herr_t HDF5CopyResizedDataset(hid_t group_id, const char *link_name,
                                     const H5L_info_t *link_info,
                                     void *extra_data) {

  my_statusmessage("Copying dataset \"%s\".", link_name);
  // recover the HDF5DatasetCopyData object
  HDF5DatasetCopyData *data =
      reinterpret_cast<HDF5DatasetCopyData *>(extra_data);
  // open the old dataset
  const hid_t old_dset = H5Dopen(group_id, link_name, H5P_DEFAULT);
  if (old_dset < 0) {
    my_errormessage("Error opening dataset \"%s\"!", link_name);
    return -1;
  }
  // get the data type
  const hid_t dtype = H5Dget_type(old_dset);
  if (dtype < 0) {
    my_errormessage("Error getting dataset datatype!");
    return -1;
  }
  // get the size (in bytes) of the data type
  const size_t dtype_size = H5Tget_size(dtype);
  if (dtype_size == 0) {
    my_errormessage("Error getting datatype size!");
    return -1;
  }
  // get the creation properties (e.g. compression options) for the old dataset
  const hid_t dprops = H5Dget_create_plist(old_dset);
  if (dprops < 0) {
    my_errormessage("Error getting dataset creation properties!");
    return -1;
  }
  // get the old data space
  const hid_t old_dspace = H5Dget_space(old_dset);
  if (old_dspace < 0) {
    my_errormessage("Error getting dataset dataspace!");
    return -1;
  }
  // query the dimensionality of the old dataset
  int ndim = H5Sget_simple_extent_ndims(old_dspace);
  if (ndim < 0) {
    my_errormessage("Error quering dataspace dimensions!");
    return -1;
  }
  // no need to limit this, but SWIFT does not use datasets with higher
  // dimensions than 2 and this allows us to hardcode some array sizes below
  if (ndim > 2) {
    my_errormessage("Cannot handle dataset with more than 2 dimensions (yet)!");
    return -1;
  }
  // query the extents of the old dataset
  hsize_t old_dims[2] = {0, 0};
  ndim = H5Sget_simple_extent_dims(old_dspace, old_dims, nullptr);
  if (ndim < 0) {
    my_errormessage("Error quering dataspace dimensions!");
    return -1;
  }
  // limit the chunk size we use for processing to the size of the dataset,
  // since we obviously cannot read more elements than there are
  hsize_t chunksize = old_dims[0];
  if (chunksize > data->get_max_chunksize()) {
    chunksize = data->get_max_chunksize();
  }
  // create a new data space for the new dataset that possibly has less elements
  hsize_t new_dims[2] = {0, 0};
  new_dims[0] = data->get_new_size();
  new_dims[1] = old_dims[1];
  const hid_t new_dspace = H5Screate_simple(ndim, new_dims, nullptr);
  if (new_dspace < 0) {
    my_errormessage("Error creating dataspace for dataset \"%s\"!", link_name);
    return -1;
  }
  // figure out if the original dataset had a chunked layout
  const hid_t layout = H5Pget_layout(dprops);
  if (layout < 0) {
    my_errormessage("Error retrieving dataset layout!");
    return -1;
  }
  if (layout == H5D_CHUNKED) {
    // if the original dataset was chunked, then the new dataset will be so as
    // well
    // however, the chunk size should be limited to the number of elements in
    // the new dataset, since this can be smaller than the original number of
    // elements
    // creating a dataset with a chunk size larger than the number of elements
    // in the dataset leads to errors (as I discovered the hard way)
    ndim = H5Pget_chunk(dprops, ndim, new_dims);
    if (ndim < 0) {
      my_errormessage("Error retrieving chunk shape for data!");
      return -1;
    }
    // note that we overwrite the new_dims array here, so we don't want to use
    // it below (at least not to represent the dimensions of the new dataset
    if (new_dims[0] > data->get_new_size()) {
      my_statusmessage("Changing chunk shape!");
      new_dims[0] = data->get_new_size();
      const herr_t hdf5status = H5Pset_chunk(dprops, ndim, new_dims);
      if (hdf5status < 0) {
        my_errormessage("Error changing chunk shape!");
        return -1;
      }
    }
  }
  // create the new dataset, using the new data space (with potentially less
  // elements) and using the same data type and dataset properties as the old
  // dataset (with a potential change to the chunk size if required)
  const hid_t new_dset =
      H5Dcreate(data->get_output_group(), link_name, dtype, new_dspace,
                H5P_DEFAULT, dprops, H5P_DEFAULT);
  if (new_dset < 0) {
    my_errormessage("Error creating dataset \"%s\"!", link_name);
    return -1;
  }
  // copy the dataset attributes
  CopyAttributes(old_dset, new_dset);
  // now copy the actual data
  // we use chunks to limit the memory footprint
  // each chunk is read from the old dataset and then written to the new dataset
  const hsize_t nchunk =
      old_dims[0] / chunksize + (old_dims[0] % chunksize > 0);
  // we need to keep track of the offsets and sizes of the newly written chunks,
  // since these are not guaranteed to have a fixed size (elements are filtered
  // out)
  hsize_t new_chunk_ofs[2] = {0, 0};
  hsize_t new_chunk_siz[2] = {0, 1};
  if (ndim > 1) {
    new_chunk_siz[1] = old_dims[1];
  }
  for (hsize_t ichunk = 0; ichunk < nchunk; ++ichunk) {
    const hsize_t dim1 = (ndim > 1) ? old_dims[1] : 1;
    const hsize_t chunk_ofs[2] = {ichunk * chunksize, 0};
    const hsize_t chunk_end = std::min((ichunk + 1) * chunksize, old_dims[0]);
    const hsize_t chunk_siz[2] = {chunk_end - chunk_ofs[0], dim1};
    // since we have no way of knowing the (general) data type of a dataset,
    // we have to read and write the data as a byte buffer
    // this is not a problem, since the HDF5 library does know the data type
    // and still handles it appropriately
    std::vector<char> buffer(dtype_size * chunk_siz[0] * chunk_siz[1]);
    data->register_size(buffer.size());
    // select the chunk in the old data space
    herr_t hdf5status = H5Sselect_hyperslab(
        old_dspace, H5S_SELECT_SET, chunk_ofs, nullptr, chunk_siz, nullptr);
    if (hdf5status < 0) {
      my_errormessage("Error selecting chunk in dataset \"%s\"!", link_name);
      return -1;
    }
    // create a memory space that can hold the chunk
    const hid_t memspace_read = H5Screate_simple(ndim, chunk_siz, nullptr);
    if (memspace_read < 0) {
      my_errormessage("Failed to allocate memory space!");
      return -1;
    }
    // read the chunk
    hdf5status = H5Dread(old_dset, dtype, memspace_read, old_dspace,
                         H5P_DEFAULT, &buffer[0]);
    if (hdf5status < 0) {
      my_errormessage("Error reading dataset \"%s\"!", link_name);
      return -1;
    }
    hdf5status = H5Sclose(memspace_read);
    if (hdf5status < 0) {
      my_errormessage("Error closing memory space!");
      return -1;
    }
    // copy over all elements in the buffer that need to be kept
    // we keep a separate count for the remaining elements
    hsize_t newi = 0;
    for (hsize_t i = 0; i < chunk_siz[0]; ++i) {
      if (data->keep(chunk_ofs[0] + i)) {
        for (hsize_t j = 0; j < chunk_siz[1]; ++j) {
          for (size_t k = 0; k < dtype_size; ++k) {
            buffer[newi * dim1 * dtype_size + j * dtype_size + k] =
                buffer[i * dim1 * dtype_size + j * dtype_size + k];
          }
        }
        ++newi;
      }
    }
    new_chunk_siz[0] = newi;
    // if any elements are left, write them to the new dataset
    if (new_chunk_siz[0] > 0) {
      // select the appropriate chunk in the new data space
      hdf5status =
          H5Sselect_hyperslab(new_dspace, H5S_SELECT_SET, new_chunk_ofs,
                              nullptr, new_chunk_siz, nullptr);
      if (hdf5status < 0) {
        my_errormessage("Error selecting chunk in dataset \"%s\"!", link_name);
        return -1;
      }
      // create a memory space large enough for the write
      const hid_t memspace_write =
          H5Screate_simple(ndim, new_chunk_siz, nullptr);
      if (memspace_write < 0) {
        my_errormessage("Failed to allocate memory space!");
        return -1;
      }
      // now write the new chunk to the new dataset
      hdf5status = H5Dwrite(new_dset, dtype, memspace_write, new_dspace,
                            H5P_DEFAULT, &buffer[0]);
      if (hdf5status < 0) {
        my_errormessage("Error writing dataset \"%s\"!", link_name);
        return -1;
      }
      hdf5status = H5Sclose(memspace_write);
      if (hdf5status < 0) {
        my_errormessage("Error closing memory space!");
        return -1;
      }
    }
    // update the offset counter in the new data space
    new_chunk_ofs[0] += new_chunk_siz[0];
  }
  // close *everything*
  herr_t hdf5status = H5Sclose(new_dspace);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataspace!");
    return -1;
  }
  hdf5status = H5Dclose(new_dset);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataset \"%s\"!", link_name);
    return -1;
  }
  hdf5status = H5Pclose(dprops);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataset creation properties!");
    return -1;
  }
  hdf5status = H5Sclose(old_dspace);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataspace!");
    return -1;
  }
  hdf5status = H5Tclose(dtype);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataset datatype!");
    return -1;
  }
  hdf5status = H5Dclose(old_dset);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataset \"%s\"!", link_name);
    return -1;
  }
  return 0;
}

/**
 * Get the last part of a file path, i.e. the path without any
 * leading directory parts.
 *
 * Example:
 *   /path/to/file.txt --> file.txt
 *   ../../weird/relative/path.hdf5 --> path.hdf5
 *   no_folders.yml --> no_folders.yml
 *
 * @param fullname Full absolute or relative file path.
 * @return Last part of path, without any folders.
 */
inline std::string get_basename(const std::string fullname) {
  size_t pos = fullname.find_last_of("/");
  if (pos != std::string::npos) {
    return fullname.substr(pos + 1);
  } else {
    return fullname;
  }
}

/**
 * @brief Copy all datasets from the source group into the destination group,
 * filtering out elements that do not need to be copied.
 *
 * @param input_group Source file/group handle.
 * @param output_group Destination file/group handle.
 * @param new_size Number of elements in the new datasets, i.e. number of
 * elements in mask that are set to true.
 * @param mask Mask that specifies which elements in the old datasets are to be
 * written to the new datasets.
 * @param max_chunksize Maximum allowed number of elements that can be
 * read/written in one go. The default value is set to -1 to force overflow and
 * initialisation to the largest possible size_t.
 * @return Maximum size (in bytes) of any buffer used during the copy operation.
 */
inline size_t CopyDatasets(const HDF5FileOrGroup input_group,
                           const HDF5FileOrGroup output_group,
                           const size_t new_size, const std::vector<bool> &mask,
                           const size_t max_chunksize = -1) {

  HDF5DatasetCopyData copy_data(output_group, new_size, mask, max_chunksize);
  const herr_t hdf5status =
      H5Literate(input_group, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr,
                 HDF5CopyResizedDataset, &copy_data);
  if (hdf5status < 0) {
    my_error("Error during dataset copy iteration!");
  }
  return copy_data.get_copy_size();
}

/**
 * @brief Auxiliary object used by HDF5CopyVirtualDataset() and
 * CopyVirtualDatasets() to copy over virtual datasets from one file to
 * another.
 */
class HDF5VirtualDatasetCopyData {
private:
  /*! @brief Destination file/group handle. */
  const HDF5FileOrGroup _output_group;
  /*! @brief Path to the group in the real files. */
  const std::string _input_group_path;
  /*! @brief Offsets of the real file data in the virtual datasets. */
  const std::vector<int64_t> &_file_offsets;
  /*! @brief Names of the real files. */
  const std::vector<std::string> &_file_names;

public:
  /**
   * @brief Constructor.
   *
   * @param output_group Destination file/group handle.
   * @param input_group_path Path to the group in the real files.
   * @param file_offsets Offsets of the real file data in the virtual datasets.
   * @param file_names Names of the real files.
   */
  inline HDF5VirtualDatasetCopyData(const HDF5FileOrGroup output_group,
                                    const std::string input_group_path,
                                    const std::vector<int64_t> &file_offsets,
                                    const std::vector<std::string> &file_names)
      : _output_group(output_group), _input_group_path(input_group_path),
        _file_offsets(file_offsets), _file_names(file_names) {}

  /**
   * @brief Get the destination file/group handle.
   *
   * @return Destination file/group handle.
   */
  inline HDF5FileOrGroup get_output_group() const { return _output_group; }

  /**
   * @brief Get the path to the group in the real files.
   *
   * @return Name of the group that holds the source datasets in the real
   * files.
   */
  inline std::string get_input_group_path() const { return _input_group_path; }

  /**
   * @brief Get the size of the new virtual datasets.
   *
   * @return Number of elements in the new virtual datasets.
   */
  inline hsize_t get_new_size() const { return _file_offsets.back(); }

  /**
   * @brief Get the offset of the file with the given index in the new
   * virtual datasets.
   *
   * @param i Real file index.
   * @return Offset of the real file data in the virtual datasets.
   */
  inline hsize_t get_file_offset(const size_t i) const {
    return _file_offsets[i];
  }

  /**
   * @brief Get the name of the file with the given index.
   *
   * @param i Real file index.
   * @return Real file name (path relative to the new virtual file).
   */
  inline std::string get_file_name(const size_t i) const {
    return _file_names[i];
  }

  /**
   * @brief Get the basename (without any folders) of the file with the given
   * index.
   *
   * @param i Real file index.
   * @return Real file basename.
   */
  inline std::string get_base_file_name(const size_t i) const {
    return get_basename(_file_names[i]);
  }

  /**
   * @brief Get the number of real files.
   *
   * @return Number of files that contain source data that should be linked in
   * the virtual datasets.
   */
  inline size_t get_number_of_files() const { return _file_names.size(); }
};

/**
 * @brief H5Literate() compatible copy function that copies a virtual dataset
 * from one file to another, adjusting the virtual links appropriately.
 *
 * @param group_id Source file/group handle.
 * @param link_name Name of the dataset currently being looked at.
 * @param link_info Additional information about the dataset currently being
 * looked at.
 * @param extra_data Void* casted HDF5VirtualDatasetCopyData object.
 * @return H5Literate() compatible return value: 0 if the object was
 * successfully processed and we can continue with the next iteration, -1 if
 * an error occurred and the loop should be aborted, 1 if we want to stop
 * the iteration early (which we do not do here).
 */
inline herr_t HDF5CopyVirtualDataset(hid_t group_id, const char *link_name,
                                     const H5L_info_t *link_info,
                                     void *extra_data) {

  my_statusmessage("Copying dataset \"%s\".", link_name);
  // extract the HDF5VirtualDatasetCopyData object
  HDF5VirtualDatasetCopyData *data =
      reinterpret_cast<HDF5VirtualDatasetCopyData *>(extra_data);
  // open the old dataset
  const hid_t old_dset = H5Dopen(group_id, link_name, H5P_DEFAULT);
  if (old_dset < 0) {
    my_errormessage("Error opening dataset \"%s\"!", link_name);
    return -1;
  }
  // open the old data space
  const hid_t old_dspace = H5Dget_space(old_dset);
  if (old_dspace < 0) {
    my_errormessage("Error getting dataset dataspace!");
    return -1;
  }
  // query the old data space dimensionality
  int ndim = H5Sget_simple_extent_ndims(old_dspace);
  if (ndim < 0) {
    my_errormessage("Error quering dataspace dimensions!");
    return -1;
  }
  // this is strictly speaking not necessary, but since SWIFT only uses up to
  // 2D datasets, it allows us to hardcode some array sizes below
  if (ndim > 2) {
    my_errormessage("Cannot handle dataset with more than 2 dimensions (yet)!");
    return -1;
  }
  // query the old data space extents
  hsize_t old_dims[2] = {0, 0};
  ndim = H5Sget_simple_extent_dims(old_dspace, old_dims, nullptr);
  if (ndim < 0) {
    my_errormessage("Error quering dataspace dimensions!");
    return -1;
  }
  herr_t hdf5status = H5Sclose(old_dspace);
  if (hdf5status < 0) {
    my_errormessage("Error closing old dataspace for dataset \"%s\"!",
                    link_name);
    return -1;
  }
  // get the old data type
  const hid_t dtype = H5Dget_type(old_dset);
  if (dtype < 0) {
    my_errormessage("Error getting dataset datatype!");
    return -1;
  }
  // create a new data space for the new virtual dataset, using the old
  // dimensionality but the new number of rows
  const hsize_t dims[2] = {data->get_new_size(), old_dims[1]};
  const hid_t filespace = H5Screate_simple(ndim, dims, nullptr);
  if (filespace < 0) {
    my_errormessage("Failed to create dataspace for dataset \"%s\"!",
                    link_name);
    return -1;
  }

  // create the new virtual data links, one for each real file
  const hid_t prop = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t offset[2] = {0, 0};
  hsize_t size[2] = {0, old_dims[1]};
  for (size_t ifile = 0; ifile < data->get_number_of_files(); ++ifile) {
    size[0] = data->get_file_offset(ifile) - offset[0];
    // select the appropriate hyperslab in the new dataset that corresponds
    // to the data in the real file
    hdf5status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr,
                                     size, nullptr);
    if (hdf5status < 0) {
      my_errormessage(
          "Failed to select hyperslab in virtual dataspace for dataset \"%s\"!",
          link_name);
      return -1;
    }
    // create a file space corresponding to the dimensions of the real file
    // data space
    const hid_t source_space = H5Screate_simple(ndim, size, nullptr);
    if (source_space < 0) {
      my_errormessage(
          "Failed to create virtual dataspace for dataset \"%s\" in file "
          "\"%s\"!",
          link_name, data->get_file_name(ifile).c_str());
      return -1;
    }
    // set the path to the real dataset in the real file
    std::stringstream dset_source;
    dset_source << data->get_input_group_path() << "/" << link_name;
    // now create the virtual link
    hdf5status =
        H5Pset_virtual(prop, filespace, data->get_base_file_name(ifile).c_str(),
                       dset_source.str().c_str(), source_space);
    if (hdf5status < 0) {
      my_errormessage(
          "Failed to create virtual link to file \"%s\" for dataset \"%s\"!",
          data->get_file_name(ifile).c_str(), link_name);
      return -1;
    }
    hdf5status = H5Sclose(source_space);
    if (hdf5status < 0) {
      my_errormessage(
          "Failed to close virtual dataspace for dataset \"%s\" in file "
          "\"%s\"!",
          link_name, data->get_file_name(ifile).c_str());
      return -1;
    }
    // update the virtual offset within the new dataset
    offset[0] += size[0];
    my_assert(offset[0] == data->get_file_offset(ifile), "Offset mismatch!");
  }
  // create the new dataset
  const hid_t new_dset = H5Dcreate(data->get_output_group(), link_name, dtype,
                                   filespace, H5P_DEFAULT, prop, H5P_DEFAULT);
  if (new_dset < 0) {
    my_errormessage("Error creating dataset \"%s\"!", link_name);
    return -1;
  }
  // copy the dataset attributes between the old and new dataset
  CopyAttributes(old_dset, new_dset);
  // close everything...
  hdf5status = H5Sclose(filespace);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataspace!");
    return -1;
  }
  hdf5status = H5Dclose(new_dset);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataset \"%s\"!", link_name);
    return -1;
  }
  hdf5status = H5Pclose(prop);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataset creation properties!");
    return -1;
  }
  hdf5status = H5Tclose(dtype);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataset datatype!");
    return -1;
  }
  hdf5status = H5Dclose(old_dset);
  if (hdf5status < 0) {
    my_errormessage("Error closing dataset \"%s\"!", link_name);
    return -1;
  }
  return 0;
}

/**
 * @brief Copy all virtual datasets from the given input file/group to the given
 * output file/group, using the given real file names and offsets to adjust the
 * virtual links.
 *
 * Note our (somewhat weird) convention that the file_offsets are shifted by
 * one element. We do this because the offset for file 0 is trivially 0, while
 * the offset beyond the last element corresponds to the total size of the new
 * datasets, which is actually useful information.
 *
 * If the virtual data links to 4 real files that contain N1...N4 elements each,
 * then the file_offsets and file_names should be as follows:
 *  file_offsets[4] = {N1, N1+N2, N1+N2+N3, N1+N2+N3+N4}
 *  file_names[4] = {"file1", "file2", "file3", "file4"}
 *
 * @param input_group Source file/group handle.
 * @param output_group Destination file/group handle.
 * @param input_group_path Name of the source group in the real files. E.g. if
 * you are copying virtual datasets for the PartType0 group, this should be
 * "PartType0".
 * @param file_offsets Offsets of the individual real file data in the new
 * virtual datasets. In fact, the offset for file i should be the offset for
 * file i+1, so that the last offset is the total number of elements in the
 * dataset (because the first offset is trivially known to be 0).
 * @param file_names Names of the real files.
 */
inline void CopyVirtualDatasets(const HDF5FileOrGroup input_group,
                                const HDF5FileOrGroup output_group,
                                const std::string input_group_path,
                                const std::vector<int64_t> &file_offsets,
                                const std::vector<std::string> &file_names) {

  // create an HDF5VirtualDatasetCopyData object to pass on additional arguments
  // to HDF5CopyVirtualDataset()
  HDF5VirtualDatasetCopyData copy_data(output_group, input_group_path,
                                       file_offsets, file_names);
  // now iterate over all datasets in the source group
  const herr_t hdf5status =
      H5Literate(input_group, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr,
                 HDF5CopyVirtualDataset, &copy_data);
  if (hdf5status < 0) {
    my_error("Error during dataset copy iteration!");
  }
}

/**
 * @brief Write a new dataset to an HDF5 file or group.
 *
 * Using a dataset name that already exists will result in an error.
 *
 * @tparam _type_ Data type of the values.
 * @param group File/group handle.
 * @param name Name of the new dataset.
 * @param data Data to write. The size of the dataset is based on the size of
 * this vector.
 */
template <typename _type_>
inline void WriteDataset(const HDF5FileOrGroup group, const std::string name,
                         const std::vector<_type_> &data) {

  // get the HDF5 data type for _type_
  const hid_t datatype = HDF5Datatypes::get_datatype_name<_type_>();

  // create the data space for the new dataset
  // use a chunked layout with a default chunk size of 2^10
  const hsize_t vsize = data.size();
  const hsize_t limit = 1 << 10;
  const hsize_t dims[1] = {vsize};
  const hsize_t chunk[1] = {std::min(vsize, limit)};
  const hid_t filespace = H5Screate_simple(1, dims, nullptr);
  if (filespace < 0) {
    my_error("Failed to create dataspace for dataset \"%s\"!", name.c_str());
  }

  // enable maximal gzip data compression and enable the Fletcher32 checksum
  // filter and the shuffle filter
  const hid_t prop = H5Pcreate(H5P_DATASET_CREATE);
  herr_t hdf5status = H5Pset_chunk(prop, 1, chunk);
  if (hdf5status < 0) {
    my_error("Failed to set chunk size for dataset \"%s\"", name.c_str());
  }
  hdf5status = H5Pset_fletcher32(prop);
  if (hdf5status < 0) {
    my_error("Failed to set Fletcher32 filter for dataset \"%s\"",
             name.c_str());
  }
  hdf5status = H5Pset_shuffle(prop);
  if (hdf5status < 0) {
    my_error("Failed to set shuffle filter for dataset \"%s\"", name.c_str());
  }
  hdf5status = H5Pset_deflate(prop, 9);
  if (hdf5status < 0) {
    my_error("Failed to set compression for dataset \"%s\"", name.c_str());
  }

  // create the dataset
  const hid_t dataset = H5Dcreate(group, name.c_str(), datatype, filespace,
                                  H5P_DEFAULT, prop, H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Failed to create dataset \"%s\"", name.c_str());
  }

  // write the data to the dataset
  hdf5status =
      H5Dwrite(dataset, datatype, H5S_ALL, filespace, H5P_DEFAULT, &data[0]);
  if (hdf5status < 0) {
    my_error("Failed to write dataset \"%s\"", name.c_str());
  }

  // close creation properties
  hdf5status = H5Pclose(prop);
  if (hdf5status < 0) {
    my_error("Failed to close creation properties for dataset \"%s\"",
             name.c_str());
  }

  // close dataspace
  hdf5status = H5Sclose(filespace);
  if (hdf5status < 0) {
    my_error("Failed to close dataspace of dataset \"%s\"", name.c_str());
  }

  // close dataset
  hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Failed to close dataset \"%s\"", name.c_str());
  }
}

/**
 * @brief Write a new virtual dataset to an HDF5 file or group.
 *
 * Using a dataset name that already exists will result in an error.
 *
 * The syntax of this function is equivalent to that of CopyVirtualDatasets().
 *
 * @tparam _type_ Data type of the values.
 * @param group File/group handle.
 * @param name Name of the new dataset.
 * @param input_group_path Name of the group that contains the dataset in the
 * real files we link to.
 * @param file_offsets Adjusted offsets of the dataset values in the real files
 * (see documentation for CopyVirtualDatasets()).
 * @param file_names Real file names.
 */
template <typename _type_>
inline void WriteVirtualDataset(const HDF5FileOrGroup group,
                                const std::string name,
                                const std::string input_group_path,
                                const std::vector<int64_t> &file_offsets,
                                const std::vector<std::string> &file_names) {

  // get the HDF5 data type for _type_
  const hid_t datatype = HDF5Datatypes::get_datatype_name<_type_>();

  // create the virtual dataspace
  const hsize_t vsize = file_offsets.back();
  const hsize_t dims[1] = {vsize};
  const hid_t filespace = H5Screate_simple(1, dims, nullptr);
  if (filespace < 0) {
    my_error("Failed to create dataspace for dataset \"%s\"!", name.c_str());
  }

  // set up the virtual data links
  const hid_t prop = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t offset[1] = {0};
  hsize_t size[1] = {0};
  for (size_t ifile = 0; ifile < file_names.size(); ++ifile) {
    size[0] = file_offsets[ifile] - offset[0];
    herr_t hdf5status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset,
                                            nullptr, size, nullptr);
    if (hdf5status < 0) {
      my_error(
          "Failed to select hyperslab in virtual dataspace for dataset \"%s\"!",
          name.c_str());
    }
    const hid_t source_space = H5Screate_simple(1, size, nullptr);
    if (source_space < 0) {
      my_error("Failed to create virtual dataspace for dataset \"%s\" in file "
               "\"%s\"!",
               name.c_str(), file_names[ifile].c_str());
    }
    std::stringstream dset_source;
    dset_source << input_group_path << "/" << name;
    hdf5status =
        H5Pset_virtual(prop, filespace, get_basename(file_names[ifile]).c_str(),
                       dset_source.str().c_str(), source_space);
    if (hdf5status < 0) {
      my_error(
          "Failed to create virtual link to file \"%s\" for dataset \"%s\"!",
          file_names[ifile].c_str(), name.c_str());
    }
    hdf5status = H5Sclose(source_space);
    if (hdf5status < 0) {
      my_error("Failed to close virtual dataspace for dataset \"%s\" in file "
               "\"%s\"!",
               name.c_str(), file_names[ifile].c_str());
    }
    offset[0] += size[0];
    my_assert(static_cast<int64_t>(offset[0]) == file_offsets[ifile],
              "Offset mismatch!");
  }

  // create the dataset
  const hid_t dataset = H5Dcreate(group, name.c_str(), datatype, filespace,
                                  H5P_DEFAULT, prop, H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Failed to create dataset \"%s\"", name.c_str());
  }

  // close creation properties
  herr_t hdf5status = H5Pclose(prop);
  if (hdf5status < 0) {
    my_error("Failed to close creation properties for dataset \"%s\"",
             name.c_str());
  }

  // close dataspace
  hdf5status = H5Sclose(filespace);
  if (hdf5status < 0) {
    my_error("Failed to close dataspace of dataset \"%s\"", name.c_str());
  }

  // close dataset
  hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Failed to close dataset \"%s\"", name.c_str());
  }
}

/**
 * @brief Replace the dataset with the given name using the given new data
 * values.
 *
 * @tparam _type_ Data value type.
 * @param file File/group handle.
 * @param name Name of the dataset (should exist).
 * @param value New values (should match the dimensions of the original
 * dataset).
 */
template <typename _type_>
inline void ReplaceDataset(const HDF5FileOrGroup file, const std::string name,
                           std::vector<_type_> &value) {

  // get the HDF5 data type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();

  // open the (existing) dataset
  const hid_t dataset = H5Dopen(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Unable to open dataset \"%s\"!", name.c_str());
  }
  // overwrite the data
  herr_t hdf5status =
      H5Dwrite(dataset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value[0]);
  if (hdf5status < 0) {
    my_error("Error writing dataset \"%s\"!", name.c_str());
  }
  hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Error closing dataset!");
  }
}

/**
 * @brief Open the HDF5 dataset with the given name.
 *
 * The dataset should be closed using CloseDataset() when it is no longer
 * needed.
 *
 * @param group File/group handle that contains the dataset.
 * @param name Name of the dataset (should exist).
 * @return Handle to the dataset that can be used in consecutive operations.
 */
inline HDF5FileOrGroup OpenDataset(const HDF5FileOrGroup group,
                                   const std::string name) {
  const hid_t dataset = H5Dopen(group, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    my_error("Unable to open dataset!");
  }
  return dataset;
}

/**
 * @brief Close an HDF5 dataset.
 *
 * After the dataset has been closed, subsequent operations using the dataset
 * handle will result in errors.
 *
 * @param dataset Existing dataset handle.
 */
inline void CloseDataset(const HDF5FileOrGroup dataset) {
  const herr_t hdf5status = H5Dclose(dataset);
  if (hdf5status < 0) {
    my_error("Error closing dataset!");
  }
}

/**
 * @brief Add a scalar attribute to a group/dataset.
 *
 * @tparam _type_ Type of the attribute.
 * @param dataset Group/dataset handle to add the attribute to.
 * @param name Name of the attribute.
 * @param value Value for the attribute.
 */
template <typename _type_>
inline void AddAttribute(const HDF5FileOrGroup dataset, const std::string name,
                         const _type_ value) {

  // get the HDF5 data type for _type_
  const hid_t dtype = HDF5Datatypes::get_datatype_name<_type_>();

  // create the attribute data space
  const hid_t aspace = H5Screate(H5S_SIMPLE);
  if (aspace < 0) {
    my_error("Error while creating dataspace for attribute \"%s\"!",
             name.c_str());
  }

  // set the attribute dimensionality to that of a scalar
  const hsize_t dim[1] = {1};
  herr_t hdf5status = H5Sset_extent_simple(aspace, 1, dim, nullptr);
  if (hdf5status < 0) {
    my_error("Error while changing dataspace shape for attribute \"%s\"!",
             name.c_str());
  }

  // create the attribute
  const hid_t attr =
      H5Acreate(dataset, name.c_str(), dtype, aspace, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0) {
    my_error("Error while creating attribute \"%s\"!", name.c_str());
  }

  // write the attribute value
  hdf5status = H5Awrite(attr, dtype, &value);
  if (hdf5status < 0) {
    my_error("Error while reading attribute \"%s\"!", name.c_str());
  }

  // close everything
  hdf5status = H5Sclose(aspace);
  if (hdf5status < 0) {
    my_error("Error closing attribute dataspace!");
  }
  hdf5status = H5Aclose(attr);
  if (hdf5status < 0) {
    my_error("Error closing attribute \"%s\"!", name.c_str());
  }
}

/**
 * @brief Add a scalar string attribute to a group/dataset.
 *
 * AddAttribute() specialisation for a C-style string.
 *
 * @param dataset Group/dataset handle to add the attribute to.
 * @param name Name of the attribute.
 * @param value Value for the attribute.
 */
template <>
inline void AddAttribute<const char *>(const HDF5FileOrGroup dataset,
                                       const std::string name,
                                       const char *value) {

  // determine the length of the string
  const int length = std::strlen(value);
  // create a scalar data space for the attribute
  const hid_t aspace = H5Screate(H5S_SCALAR);
  if (aspace < 0) {
    my_error("Error while creating dataspace for attribute \"%s\"!",
             name.c_str());
  }

  // create a new data type that is a copy of the default length 1 C-string
  // data type
  const hid_t dtype = H5Tcopy(H5T_C_S1);
  if (dtype < 0) {
    my_error("Error while copying datatype \"H5T_C_S1\"!");
  }

  // now adjust the length of the new data type to that of the actual string
  herr_t hdf5status = H5Tset_size(dtype, length);
  if (hdf5status < 0) {
    my_error("Error while resizing attribute type to length %i!", length);
  }

  // create the attribute
  const hid_t attr =
      H5Acreate(dataset, name.c_str(), dtype, aspace, H5P_DEFAULT, H5P_DEFAULT);
  if (attr < 0) {
    my_error("Error while creating attribute \"%s\"!", name.c_str());
  }

  // write the attribute value
  hdf5status = H5Awrite(attr, dtype, value);
  if (hdf5status < 0) {
    my_error("Error while reading attribute \"%s\"!", name.c_str());
  }

  // close everything
  hdf5status = H5Tclose(dtype);
  if (hdf5status < 0) {
    my_error("Error closing datatype!");
  }
  hdf5status = H5Sclose(aspace);
  if (hdf5status < 0) {
    my_error("Error closing attribute dataspace!");
  }
  hdf5status = H5Aclose(attr);
  if (hdf5status < 0) {
    my_error("Error closing attribute \"%s\"!", name.c_str());
  }
}

/**
 * @brief Check if a group/dataset has an attribute with the given name.
 *
 * @param group Group/dataset handle.
 * @param name Name of an attribute.
 * @return True if the group/dataset has an attribute with that name.
 */
inline bool AttributeExists(const HDF5FileOrGroup group,
                            const std::string name) {
  // check if the attribute exists
  htri_t hdf5status = H5Aexists(group, name.c_str());
  if (hdf5status < 0) {
    my_error("Error while trying to check existence of attribute \"%s\"!",
             name.c_str());
  }
  // convert the htri_t return value to a boolean
  return hdf5status > 0;
}

/**
 * @brief Check if a group/dataset exists with the given name within the given
 * file/group.
 *
 * @param group File/group handle.
 * @param name Name of a group or dataset.
 * @return True if the group/dataset exists.
 */
inline bool GroupOrDatasetExists(const HDF5FileOrGroup group,
                                 const std::string name) {
  htri_t hdf5status = H5Lexists(group, name.c_str(), H5P_DEFAULT);
  if (hdf5status < 0) {
    my_error("Error while trying to check existence of link \"%s\"!",
             name.c_str());
  }
  // convert the htri_t return value to a boolean
  return hdf5status > 0;
}

} // namespace HDF5

#endif // RS_HDF5_HPP
