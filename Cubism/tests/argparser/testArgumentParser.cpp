// File       : testArgumentParser.cpp
// Created    : Thu Mar 28 2019 05:27:31 PM (+0100)
// Author     : Fabian Wermelinger
// Description: Test Cubism ArgumentParser
// Copyright 2019 ETH Zurich. All Rights Reserved.
#include <iostream>
#include "Cubism/ArgumentParser.h"
using namespace cubism;
using namespace std;

int main(int argc, char* argv[])
{
    ArgumentParser parser(argc, argv);
    if (parser.exist("-conf"))
        parser.readFile(parser("-conf").asString());
    parser.print_args();

    cout << "s1 = " << parser("s1").asString() << endl;
    cout << "s2 = " << parser("s2").asString() << endl;

    cout << "f1 = " << parser("f1").asDouble() << endl;
    cout << "f2 = " << parser("f2").asDouble() << endl;
    cout << "f3 = " << parser("f3").asDouble() << endl;
    cout << "f4 = " << parser("f4").asDouble() << endl;
    cout << "f5 = " << parser("f5").asDouble() << endl;
    cout << "f6 = " << parser("f6").asDouble() << endl;
    cout << "f7 = " << parser("f7").asDouble() << endl;
    cout << "f8 = " << parser("f8").asDouble() << endl;
    cout << "f9 = " << parser("f9").asDouble() << endl;

    return 0;
}
