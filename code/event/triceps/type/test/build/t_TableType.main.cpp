
#include <utest/Utest.h>

int main(int ac, char **av)
{
	Utest driver(ac, av);


	extern Utest::Testcase nameSet;
	driver.addcase(nameSet, "nameSet");

	extern Utest::Testcase emptyTable;
	driver.addcase(emptyTable, "emptyTable");

	extern Utest::Testcase hashedIndex;
	driver.addcase(hashedIndex, "hashedIndex");

	extern Utest::Testcase badRow;
	driver.addcase(badRow, "badRow");

	extern Utest::Testcase nullRow;
	driver.addcase(nullRow, "nullRow");

	extern Utest::Testcase badIndexName;
	driver.addcase(badIndexName, "badIndexName");

	extern Utest::Testcase nullIndex;
	driver.addcase(nullIndex, "nullIndex");

	extern Utest::Testcase dupIndexName;
	driver.addcase(dupIndexName, "dupIndexName");

	extern Utest::Testcase throwOnBad;
	driver.addcase(throwOnBad, "throwOnBad");

	extern Utest::Testcase throwModInitalized;
	driver.addcase(throwModInitalized, "throwModInitalized");

	extern Utest::Testcase hashedNested;
	driver.addcase(hashedNested, "hashedNested");

	extern Utest::Testcase hashedBadField;
	driver.addcase(hashedBadField, "hashedBadField");

	extern Utest::Testcase fifoIndex;
	driver.addcase(fifoIndex, "fifoIndex");

	extern Utest::Testcase fifoIndexLimit;
	driver.addcase(fifoIndexLimit, "fifoIndexLimit");

	extern Utest::Testcase fifoIndexJumping;
	driver.addcase(fifoIndexJumping, "fifoIndexJumping");

	extern Utest::Testcase fifoIndexReverse;
	driver.addcase(fifoIndexReverse, "fifoIndexReverse");

	extern Utest::Testcase fifoBadJumping;
	driver.addcase(fifoBadJumping, "fifoBadJumping");

	extern Utest::Testcase fifoBadNested;
	driver.addcase(fifoBadNested, "fifoBadNested");

	extern Utest::Testcase sortedIndex;
	driver.addcase(sortedIndex, "sortedIndex");

	extern Utest::Testcase sortedIndexBad;
	driver.addcase(sortedIndexBad, "sortedIndexBad");

	extern Utest::Testcase sortedNested;
	driver.addcase(sortedNested, "sortedNested");

	extern Utest::Testcase aggregator;
	driver.addcase(aggregator, "aggregator");

	extern Utest::Testcase copy;
	driver.addcase(copy, "copy");

	return driver.run();
}
