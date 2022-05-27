//----------------------------------------------------------------------------------------------------------------------
///
/// \file       test_sparse.cpp
/// \brief      Тестирование задач о назначениях
/// \date       18.05.22 - создан
/// \author     Соболев А.А.
///

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_sparse

#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <vector>

#include <sparse.h>

BOOST_AUTO_TEST_CASE( MatrixDenseToCOO_1 )
{
    // https://en.wikipedia.org/wiki/Sparse_matrix
    arma::mat A = {
        { 1.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 3.0, 0.0, 4.0, 0.0, 0.0 },
        { 0.0, 0.0, 5.0, 6.0, 7.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 8.0 }
    };
    std::vector<double> COO_expected = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    std::vector<unsigned> row_expected = { 0, 0, 1, 1, 2, 2, 2, 3 };
    std::vector<unsigned> col_expected = { 0, 1, 1, 3, 2, 3, 4, 5 };

    std::vector<double> COO;
    std::vector<unsigned> row, col;

    SPML::Sparse::MatrixDenseToCOO( A, COO, row, col );
    BOOST_CHECK_EQUAL_COLLECTIONS( COO.begin(), COO.end(), COO_expected.begin(), COO_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( row.begin(), row.end(), row_expected.begin(), row_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( col.begin(), col.end(), col_expected.begin(), col_expected.end() );

    arma::mat Arestored;
    SPML::Sparse::MatrixCOOtoDense( COO, row, col, Arestored );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( A, Arestored, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( MatrixDenseToCSR_1 )
{
    // https://en.wikipedia.org/wiki/Sparse_matrix
    arma::mat A = {
        { 1.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 3.0, 0.0, 4.0, 0.0, 0.0 },
        { 0.0, 0.0, 5.0, 6.0, 7.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 8.0 }
    };
    std::vector<double>CSR_expected = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    std::vector<unsigned> first_expected = { 0, 1, 1, 3, 2, 3, 4, 5 };
    std::vector<unsigned> kk_expected = { 0, 2, 4, 7, 8 };

    std::vector<double> CSR;
    std::vector<unsigned> first, kk;

    SPML::Sparse::MatrixDenseToCSR( A, CSR, first, kk );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.begin(), CSR.end(), CSR_expected.begin(), CSR_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( first.begin(), first.end(), first_expected.begin(), first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( kk.begin(), kk.end(), kk_expected.begin(), kk_expected.end() );

    arma::mat Arestored;
    SPML::Sparse::MatrixCSRtoDense( CSR, first, kk, Arestored );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( A, Arestored, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( MatrixDenseToCSC_1 )
{
    // https://en.wikipedia.org/wiki/Sparse_matrix
    arma::mat A = {
        { 1.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 3.0, 0.0, 4.0, 0.0, 0.0 },
        { 0.0, 0.0, 5.0, 6.0, 7.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 8.0 }
    };
    std::vector<double>CSC_expected = { 1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0 };
    std::vector<unsigned> first_expected = { 0, 0, 1, 2, 1, 2, 2, 3 };
    std::vector<unsigned> kk_expected = { 0, 1, 3, 4, 6, 7, 8 };

    std::vector<double> CSC;
    std::vector<unsigned> first, kk;

    SPML::Sparse::MatrixDenseToCSC( A, CSC, first, kk );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.begin(), CSC.end(), CSC_expected.begin(), CSC_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( first.begin(), first.end(), first_expected.begin(), first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( kk.begin(), kk.end(), kk_expected.begin(), kk_expected.end() );

    arma::mat Arestored;
    SPML::Sparse::MatrixCSCtoDense( CSC, first, kk, Arestored );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( A, Arestored, "absdiff", eps ), true );
}
