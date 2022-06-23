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
#include <map>

#include <sparse.h>
#include <timing.h>

// https://en.wikipedia.org/wiki/Sparse_matrix
arma::mat A1 = {
    { 1.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 3.0, 0.0, 4.0, 0.0, 0.0 },
    { 0.0, 0.0, 5.0, 6.0, 7.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 8.0 }
};

// rowwise (default)
std::vector<double> A1_coo_val_expected = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
std::vector<int> A1_coo_row_expected = { 0, 0, 1, 1, 2, 2, 2, 3 };
std::vector<int> A1_coo_col_expected = { 0, 1, 1, 3, 2, 3, 4, 5 };
SPML::Sparse::CMatrixCOO A1_coo_expected{ A1_coo_val_expected, A1_coo_row_expected, A1_coo_col_expected };

// colwise
std::vector<double> A1_coo_val_expected_colwise = { 1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0 };
std::vector<int> A1_coo_row_expected_colwise = { 0, 0, 1, 2, 1, 2, 2, 3 };
std::vector<int> A1_coo_col_expected_colwise = { 0, 1, 1, 2, 3, 3, 4, 5 };
SPML::Sparse::CMatrixCOO A1_coo_expected_colwise{ A1_coo_val_expected_colwise, A1_coo_row_expected_colwise,
    A1_coo_col_expected_colwise };

std::vector<double> A1_csr_val_expected = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
std::vector<int> A1_csr_first_expected = { 0, 1, 1, 3, 2, 3, 4, 5 };
std::vector<int> A1_csr_kk_expected = { 0, 2, 4, 7, 8 };
SPML::Sparse::CMatrixCSR A1_csr_expected{ A1_csr_val_expected, A1_csr_first_expected, A1_csr_kk_expected };

std::vector<double> A1_csc_val_expected = { 1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0 };
std::vector<int> A1_csc_first_expected = { 0, 0, 1, 2, 1, 2, 2, 3 };
std::vector<int> A1_csc_kk_expected = { 0, 1, 3, 4, 6, 7, 8 };
SPML::Sparse::CMatrixCSC A1_csc_expected{ A1_csc_val_expected, A1_csc_first_expected, A1_csc_kk_expected };

BOOST_AUTO_TEST_SUITE( test_A1 )

BOOST_AUTO_TEST_CASE( MatrixDenseToCOO_A1_1 )
{
    std::vector<double> COO_value;
    std::vector<int> COO_row, COO_col;

    SPML::Sparse::MatrixDenseToCOO( A1, COO_value, COO_row, COO_col );
    BOOST_CHECK_EQUAL_COLLECTIONS( COO_value.begin(), COO_value.end(), A1_coo_val_expected.begin(), A1_coo_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( COO_row.begin(), COO_row.end(), A1_coo_row_expected.begin(), A1_coo_row_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( COO_col.begin(), COO_col.end(), A1_coo_col_expected.begin(), A1_coo_col_expected.end() );

    arma::mat Arestored;
    SPML::Sparse::MatrixCOOtoDense( COO_value, COO_row, COO_col, Arestored );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( A1, Arestored, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( MatrixDenseToCOO_A1_2 )
{
    SPML::Sparse::CMatrixCOO coo;
    SPML::Sparse::MatrixDenseToCOO( A1, coo );
    BOOST_CHECK_EQUAL_COLLECTIONS( coo.coo_val.begin(), coo.coo_val.end(), A1_coo_val_expected.begin(), A1_coo_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( coo.coo_row.begin(), coo.coo_row.end(), A1_coo_row_expected.begin(), A1_coo_row_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( coo.coo_col.begin(), coo.coo_col.end(), A1_coo_col_expected.begin(), A1_coo_col_expected.end() );

    arma::mat Arestored;
    SPML::Sparse::MatrixCOOtoDense( coo, Arestored );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( A1, Arestored, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( MatrixDenseToCSR_A1_1 )
{
    std::vector<double> CSR_value;
    std::vector<int> CSR_first, CSR_kk;

    SPML::Sparse::MatrixDenseToCSR( A1, CSR_value, CSR_first, CSR_kk );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR_value.begin(), CSR_value.end(), A1_csr_val_expected.begin(), A1_csr_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR_first.begin(), CSR_first.end(), A1_csr_first_expected.begin(), A1_csr_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR_kk.begin(), CSR_kk.end(), A1_csr_kk_expected.begin(), A1_csr_kk_expected.end() );

    arma::mat Arestored;
    SPML::Sparse::MatrixCSRtoDense( CSR_value, CSR_first, CSR_kk, Arestored );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( A1, Arestored, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( MatrixDenseToCSR_A1_2 )
{
    SPML::Sparse::CMatrixCSR CSR;

    SPML::Sparse::MatrixDenseToCSR( A1, CSR );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_val.begin(), CSR.csr_val.end(), A1_csr_val_expected.begin(), A1_csr_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_first.begin(), CSR.csr_first.end(), A1_csr_first_expected.begin(), A1_csr_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_kk.begin(), CSR.csr_kk.end(), A1_csr_kk_expected.begin(), A1_csr_kk_expected.end() );

    arma::mat Arestored;
    SPML::Sparse::MatrixCSRtoDense( CSR, Arestored );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( A1, Arestored, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( MatrixDenseToCSC_A1_1 )
{
    std::vector<double> CSC_value;
    std::vector<int> CSC_first, CSC_kk;

    SPML::Sparse::MatrixDenseToCSC( A1, CSC_value, CSC_first, CSC_kk );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC_value.begin(), CSC_value.end(), A1_csc_val_expected.begin(), A1_csc_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC_first.begin(), CSC_first.end(), A1_csc_first_expected.begin(), A1_csc_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC_kk.begin(), CSC_kk.end(), A1_csc_kk_expected.begin(), A1_csc_kk_expected.end() );

    arma::mat Arestored;
    SPML::Sparse::MatrixCSCtoDense( CSC_value, CSC_first, CSC_kk, Arestored );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( A1, Arestored, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( MatrixDenseToCSC_A1_2 )
{
    SPML::Sparse::CMatrixCSC CSC;

    SPML::Sparse::MatrixDenseToCSC( A1, CSC );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_val.begin(), CSC.csc_val.end(), A1_csc_val_expected.begin(), A1_csc_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_first.begin(), CSC.csc_first.end(), A1_csc_first_expected.begin(), A1_csc_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_kk.begin(), CSC.csc_kk.end(), A1_csc_kk_expected.begin(), A1_csc_kk_expected.end() );

    arma::mat Arestored;
    SPML::Sparse::MatrixCSCtoDense( CSC, Arestored );

    double eps = 1e-6;
    BOOST_CHECK_EQUAL( arma::approx_equal( A1, Arestored, "absdiff", eps ), true );
}

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSR_A1 )
{
    SPML::Sparse::CMatrixCSR CSR;
    SPML::Sparse::MatrixCOOtoCSR( A1_coo_expected, CSR );

    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_val.begin(), CSR.csr_val.end(), A1_csr_val_expected.begin(), A1_csr_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_first.begin(), CSR.csr_first.end(), A1_csr_first_expected.begin(), A1_csr_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_kk.begin(), CSR.csr_kk.end(), A1_csr_kk_expected.begin(), A1_csr_kk_expected.end() );
}

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSC_A1 )
{
    SPML::Sparse::CMatrixCSC CSC;
    SPML::Sparse::MatrixCOOtoCSC( A1_coo_expected, CSC );

    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_val.begin(), CSC.csc_val.end(), A1_csc_val_expected.begin(), A1_csc_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_first.begin(), CSC.csc_first.end(), A1_csc_first_expected.begin(), A1_csc_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_kk.begin(), CSC.csc_kk.end(), A1_csc_kk_expected.begin(), A1_csc_kk_expected.end() );
}

BOOST_AUTO_TEST_CASE( test_sorted )
{
    std::map< SPML::Sparse::CKeyCOO, double > myMap;
    int k_max = A1_coo_val_expected.size();
    for( int k = ( k_max - 1 ); k >= 0; k-- ) {
        int i = A1_coo_row_expected[k];
        int j = A1_coo_col_expected[k];
        double val = A1_coo_val_expected[k];
        myMap.insert( std::make_pair( SPML::Sparse::CKeyCOO( i, j ), val ) );
    }

    std::map< SPML::Sparse::CKeyCOO, double > myMap2;
    myMap2.insert( std::make_pair( SPML::Sparse::CKeyCOO( 2, 4 ), 7 ) );
    myMap2.insert( std::make_pair( SPML::Sparse::CKeyCOO( 1, 3 ), 4 ) );
    myMap2.insert( std::make_pair( SPML::Sparse::CKeyCOO( 0, 1 ), 2 ) );
    myMap2.insert( std::make_pair( SPML::Sparse::CKeyCOO( 3, 5 ), 8 ) );
    myMap2.insert( std::make_pair( SPML::Sparse::CKeyCOO( 2, 3 ), 6 ) );
    myMap2.insert( std::make_pair( SPML::Sparse::CKeyCOO( 0, 0 ), 1 ) );
    myMap2.insert( std::make_pair( SPML::Sparse::CKeyCOO( 2, 2 ), 5 ) );
    myMap2.insert( std::make_pair( SPML::Sparse::CKeyCOO( 1, 1 ), 3 ) );

    SPML::Sparse::CMatrixCOO coo;
    for( auto &v : myMap ) {
        int i = ( v.first ).i();
        int j = ( v.first ).j();
        double val = v.second;
        coo.coo_val.push_back( val );
        coo.coo_row.push_back( i );
        coo.coo_col.push_back( j );
    }

    SPML::Sparse::CMatrixCSR csr1, csr2;

    SPML::Sparse::MatrixCOOtoCSR( coo, csr1, true );
    SPML::Sparse::MatrixCOOtoCSR( coo, csr2, false );

    BOOST_CHECK_EQUAL_COLLECTIONS( csr1.csr_val.begin(), csr1.csr_val.end(), csr2.csr_val.begin(), csr2.csr_val.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( csr1.csr_first.begin(), csr1.csr_first.end(), csr2.csr_first.begin(), csr2.csr_first.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( csr1.csr_kk.begin(), csr1.csr_kk.end(), csr2.csr_kk.begin(), csr2.csr_kk.end() );
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSR_FAST )
{
    SPML::Timing::CTimeKeeper tk;
    for( int i = 0; i < 20000000; i++ ) {
        SPML::Sparse::CMatrixCSR CSR;
        tk.StartTimer();
        SPML::Sparse::MatrixCOOtoCSR( A1_coo_expected, CSR, true );
        tk.EndTimer();
    }
    std::cout << "MatrixCOOtoCSR_FAST tk.TimePerOp() = " << tk.TimePerOp() * 1.0e6 << " [us]" << std::endl;

    SPML::Sparse::CMatrixCSR CSR;
    SPML::Sparse::MatrixCOOtoCSR( A1_coo_expected, CSR, true );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_val.begin(), CSR.csr_val.end(), A1_csr_val_expected.begin(), A1_csr_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_first.begin(), CSR.csr_first.end(), A1_csr_first_expected.begin(), A1_csr_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_kk.begin(), CSR.csr_kk.end(), A1_csr_kk_expected.begin(), A1_csr_kk_expected.end() );
}

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSR_SLOW )
{
    SPML::Timing::CTimeKeeper tk;
    for( int i = 0; i < 20000000; i++ ) {
        SPML::Sparse::CMatrixCSR CSR;
        tk.StartTimer();
        SPML::Sparse::MatrixCOOtoCSR( A1_coo_expected, CSR, false );
        tk.EndTimer();
    }
    std::cout << "MatrixCOOtoCSR_SLOW tk.TimePerOp() = " << tk.TimePerOp() * 1.0e6 << " [us]" << std::endl;

    SPML::Sparse::CMatrixCSR CSR;
    SPML::Sparse::MatrixCOOtoCSR( A1_coo_expected, CSR, true );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_val.begin(), CSR.csr_val.end(), A1_csr_val_expected.begin(), A1_csr_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_first.begin(), CSR.csr_first.end(), A1_csr_first_expected.begin(), A1_csr_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_kk.begin(), CSR.csr_kk.end(), A1_csr_kk_expected.begin(), A1_csr_kk_expected.end() );
}

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSC_FAST )
{
    SPML::Timing::CTimeKeeper tk;
    for( int i = 0; i < 20000000; i++ ) {
        SPML::Sparse::CMatrixCSC CSC;
        tk.StartTimer();
        SPML::Sparse::MatrixCOOtoCSC( A1_coo_expected_colwise, CSC, true );
//        SPML::Sparse::MatrixCOOtoCSC( A1_coo_expected, CSC, false );
        tk.EndTimer();
    }
    std::cout << "MatrixCOOtoCSC_FAST tk.TimePerOp() = " << tk.TimePerOp() * 1.0e6 << " [us]" << std::endl;

    SPML::Sparse::CMatrixCSC CSC;
    SPML::Sparse::MatrixCOOtoCSC( A1_coo_expected_colwise, CSC, true );
//    SPML::Sparse::MatrixCOOtoCSC( A1_coo_expected, CSC, true );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_val.begin(), CSC.csc_val.end(), A1_csc_val_expected.begin(), A1_csc_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_first.begin(), CSC.csc_first.end(), A1_csc_first_expected.begin(), A1_csc_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_kk.begin(), CSC.csc_kk.end(), A1_csc_kk_expected.begin(), A1_csc_kk_expected.end() );
}

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSC_SLOW )
{
    SPML::Timing::CTimeKeeper tk;
    for( int i = 0; i < 20000000; i++ ) {
        SPML::Sparse::CMatrixCSC CSC;
        tk.StartTimer();
        SPML::Sparse::MatrixCOOtoCSC( A1_coo_expected_colwise, CSC, false );
//        SPML::Sparse::MatrixCOOtoCSC( A1_coo_expected, CSC, false );
        tk.EndTimer();
    }
    std::cout << "MatrixCOOtoCSC_SLOW tk.TimePerOp() = " << tk.TimePerOp() * 1.0e6 << " [us]" << std::endl;

    SPML::Sparse::CMatrixCSC CSC;
//    SPML::Sparse::MatrixCOOtoCSC( A1_coo_expected_colwise, CSC, false );
    SPML::Sparse::MatrixCOOtoCSC( A1_coo_expected, CSC, false );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_val.begin(), CSC.csc_val.end(), A1_csc_val_expected.begin(), A1_csc_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_first.begin(), CSC.csc_first.end(), A1_csc_first_expected.begin(), A1_csc_first_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_kk.begin(), CSC.csc_kk.end(), A1_csc_kk_expected.begin(), A1_csc_kk_expected.end() );
}




