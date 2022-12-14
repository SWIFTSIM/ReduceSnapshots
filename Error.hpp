/**
 * @file Error.hpp
 *
 * @brief Custom error and assertions macros.
 *
 * Useful for displaying more informative error/status messages.
 */

#ifndef RS_ERROR_HPP
#define RS_ERROR_HPP

#include <cstdlib>

/*! @brief Uncomment this to activate low-level status output (will clutter the
 *  stderr). */
//#define STATUSMESSAGES

// we depend on a global MPI_rank variable to display the rank as part of
// messages
extern int MPI_rank;

/**
 * @brief Display a status message.
 *
 * Prints the given message (including printf-style format specifiers) to stderr
 * and prepends it with information about the location of the macro call.
 *
 * @param s Message to write.
 * @param ... Additional arguments for printf().
 */
#ifdef STATUSMESSAGES
#define my_statusmessage(s, ...)                                               \
  {                                                                            \
    fprintf(stderr, "%s:%s():%i: Status: ", __FILE__, __FUNCTION__, __LINE__); \
    fprintf(stderr, s "\n", ##__VA_ARGS__);                                    \
  }
#else
#define my_statusmessage(s, ...)
#endif

/**
 * @brief Write an error message to stderr.
 *
 * Prints the given message (including printf-style format specifiers) to stderr
 * and prepends it with information about the location of the macro call.
 *
 * This error macro should be used if my_error() cannot be used in a particular
 * context (for example because you are in an HDF5 handled loop).
 *
 * @param s Message to write.
 * @param ... Additional arguments for printf().
 */
#define my_errormessage(s, ...)                                                \
  {                                                                            \
    fprintf(stderr, "[%2i] %s:%s():%i: Error:\n", MPI_rank, __FILE__,          \
            __FUNCTION__, __LINE__);                                           \
    fprintf(stderr, s "\n", ##__VA_ARGS__);                                    \
  }

/**
 * @brief Write an error message to stderr and exit.
 *
 * Prints the given message (including printf-style format specifiers) to stderr
 * and prepends it with information about the location of the macro call. Then
 * exits the program using abort().
 *
 * @param s Message to write.
 * @param ... Additional arguments for printf().
 */
#define my_error(s, ...)                                                       \
  {                                                                            \
    my_errormessage(s, ##__VA_ARGS__);                                         \
    abort();                                                                   \
  }

/**
 * @brief Assertion macro.
 *
 * Evaluates the given condition and throws an error if it evaluates to false.
 *
 * @param condition Condition to check.
 * @param s Message to write upon failure of the condition.
 * @param ... Additional arguments for printf(), depending on format specifiers
 * in s.
 */
#define my_assert(condition, s, ...)                                           \
  if (!(condition)) {                                                          \
    my_error("Assertion failed: " s, ##__VA_ARGS__);                           \
  }

#endif // RS_ERROR_HPP
