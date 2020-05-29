#include "Utils.h"
#include "../../source/utils/BufferedLogger.h"

using namespace cubismup3d;

bool testBufferedLogger()
{
  // First delete the file and add AUTO_FLUSH_COUNT - 1 items.
  // The file should not be there still. Adding one more creates the file.
  const std::string tmp = "_BufferedLogger.dat";
  const int error = std::remove(tmp.c_str());
  // Removed or file did not exist.
  CUP_CHECK(error == 0 || errno == ENOENT, "Error %d\n", error);

  for (int i = 1; i <= logger.AUTO_FLUSH_COUNT; ++i)
    logger.get_stream(tmp) << i << '\n';

  FILE *f = fopen(tmp.c_str(), "r");
  // It would contain AUTO_FLUSH_COUNT - 1 frames,
  // because flushing happens before `<< i << '\n'`.
  CUP_CHECK(f == nullptr, "File should not be here still.\n");

  logger.get_stream(tmp) << logger.AUTO_FLUSH_COUNT + 1 << '\n';
  f = fopen(tmp.c_str(), "r");
  CUP_CHECK(f !=  nullptr, "File should be there now.\n");
  for (int i = 1; i <= logger.AUTO_FLUSH_COUNT; ++i) {
    int x;
    CUP_CHECK(1 == fscanf(f, "%d ", &x) && x == i,
              "Incorrect file content, got %d instead of %d\n", x, i);
  }

  // Note: The destructor of `logger` will flush the line, but at this point
  // that line is still not in the file.
  CUP_CHECK(feof(f), "Not reached the end of the file.");

  return true;
}

int main()
{
  CUP_RUN_TEST(testBufferedLogger);
}
