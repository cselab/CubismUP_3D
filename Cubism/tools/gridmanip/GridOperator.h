// File       : GridOperator.h
// Created    : Mon Jul 10 2017 12:26:52 PM (+0200)
// Author     : Fabian Wermelinger
// Description: Operator base class
// Copyright 2017 ETH Zurich. All Rights Reserved.
#ifndef GRIDOPERATOR_H_LBMHW8RU
#define GRIDOPERATOR_H_LBMHW8RU

#include "Cubism/ArgumentParser.h"
#include "Types.h"

using namespace cubism;

template <typename TGridIn, typename TGridOut, typename TBlockLab>
class GridOperator
{
public:
    GridOperator(ArgumentParser& p) : m_parser(p) {}

    virtual ~GridOperator() = default;

    virtual void
    operator()(const TGridIn &, TGridOut &, const bool verbose = true) = 0;

protected:
    ArgumentParser& m_parser;
};

#endif /* GRIDOPERATOR_H_LBMHW8RU */

