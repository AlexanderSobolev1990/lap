//----------------------------------------------------------------------------------------------------------------------
///
/// \file       benchmark_sparse.cpp
/// \brief      Тестирование задач о назначениях
///

#define BOOST_TEST_MODULE benchmark_sparse

#include <boost/test/included/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <vector>
#include <map>

#include "spm/lap/sparse.hpp"
#include "spm/timing/timing.hpp"

// https://en.wikipedia.org/wiki/Sparse_matrix
arma::mat A1 = {
    { 1.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 3.0, 0.0, 4.0, 0.0, 0.0 },
    { 0.0, 0.0, 5.0, 6.0, 7.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 8.0 }
};
//arma::mat A1 = {
//    { 1.0, 0.0, 2.0, 0.0, 0.0 },
//    { 0.0, 3.0, 4.0, 0.0, 5.0 },
//    { 0.0, 0.0, 6.0, 7.0, 0.0 }
//};

// rowwise (default)
std::vector<double> A1_coo_val_expected = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
std::vector<int> A1_coo_row_expected = { 0, 0, 1, 1, 2, 2, 2, 3 };
std::vector<int> A1_coo_col_expected = { 0, 1, 1, 3, 2, 3, 4, 5 };
SPM::Sparse::CMatrixCOO A1_coo_expected{ A1_coo_val_expected, A1_coo_row_expected, A1_coo_col_expected };

// colwise
std::vector<double> A1_coo_val_expected_colwise = { 1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0 };
std::vector<int> A1_coo_row_expected_colwise = { 0, 0, 1, 2, 1, 2, 2, 3 };
std::vector<int> A1_coo_col_expected_colwise = { 0, 1, 1, 2, 3, 3, 4, 5 };
SPM::Sparse::CMatrixCOO A1_coo_expected_colwise{ A1_coo_val_expected_colwise, A1_coo_row_expected_colwise,
    A1_coo_col_expected_colwise };

std::vector<double> A1_csr_val_expected = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
std::vector<int> A1_csr_kk_expected = { 0, 1, 1, 3, 2, 3, 4, 5 };
std::vector<int> A1_csr_first_expected = { 0, 2, 4, 7, 8 };
SPM::Sparse::CMatrixCSR A1_csr_expected{ A1_csr_val_expected, A1_csr_kk_expected, A1_csr_first_expected };

std::vector<double> A1_csc_val_expected = { 1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0 };
std::vector<int> A1_csc_kk_expected = { 0, 0, 1, 2, 1, 2, 2, 3 };
std::vector<int> A1_csc_first_expected = { 0, 1, 3, 4, 6, 7, 8 };
SPM::Sparse::CMatrixCSC A1_csc_expected{ A1_csc_val_expected, A1_csc_kk_expected, A1_csc_first_expected };

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSR_FAST )
{
    SPM::Timing::CTimeKeeper tk;
    for( int i = 0; i < 20000000; i++ ) {
        SPM::Sparse::CMatrixCSR CSR;
        tk.StartTimer();
        SPM::Sparse::MatrixCOOtoCSR( A1_coo_expected, CSR, true );
        tk.EndTimer();
    }
    std::cout << "MatrixCOOtoCSR_FAST tk.TimePerOp() = " << tk.TimePerOp() * 1.0e6 << " [us]" << std::endl;

    SPM::Sparse::CMatrixCSR CSR;
    SPM::Sparse::MatrixCOOtoCSR( A1_coo_expected, CSR, true );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_val.begin(), CSR.csr_val.end(), A1_csr_val_expected.begin(), A1_csr_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_kk.begin(), CSR.csr_kk.end(), A1_csr_kk_expected.begin(), A1_csr_kk_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_first.begin(), CSR.csr_first.end(), A1_csr_first_expected.begin(), A1_csr_first_expected.end() );
}

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSR_SLOW )
{
    SPM::Timing::CTimeKeeper tk;
    for( int i = 0; i < 20000000; i++ ) {
        SPM::Sparse::CMatrixCSR CSR;
        tk.StartTimer();
        SPM::Sparse::MatrixCOOtoCSR( A1_coo_expected, CSR, false );
        tk.EndTimer();
    }
    std::cout << "MatrixCOOtoCSR_SLOW tk.TimePerOp() = " << tk.TimePerOp() * 1.0e6 << " [us]" << std::endl;

    SPM::Sparse::CMatrixCSR CSR;
    SPM::Sparse::MatrixCOOtoCSR( A1_coo_expected, CSR, true );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_val.begin(), CSR.csr_val.end(), A1_csr_val_expected.begin(), A1_csr_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_kk.begin(), CSR.csr_kk.end(), A1_csr_kk_expected.begin(), A1_csr_kk_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSR.csr_first.begin(), CSR.csr_first.end(), A1_csr_first_expected.begin(), A1_csr_first_expected.end() );
}

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSC_FAST )
{
    SPM::Timing::CTimeKeeper tk;
    for( int i = 0; i < 20000000; i++ ) {
        SPM::Sparse::CMatrixCSC CSC;
        tk.StartTimer();
        SPM::Sparse::MatrixCOOtoCSC( A1_coo_expected_colwise, CSC, true );
        tk.EndTimer();
    }
    std::cout << "MatrixCOOtoCSC_FAST tk.TimePerOp() = " << tk.TimePerOp() * 1.0e6 << " [us]" << std::endl;

    SPM::Sparse::CMatrixCSC CSC;
    SPM::Sparse::MatrixCOOtoCSC( A1_coo_expected_colwise, CSC, true );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_val.begin(), CSC.csc_val.end(), A1_csc_val_expected.begin(), A1_csc_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_kk.begin(), CSC.csc_kk.end(), A1_csc_kk_expected.begin(), A1_csc_kk_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_first.begin(), CSC.csc_first.end(), A1_csc_first_expected.begin(), A1_csc_first_expected.end() );
}

BOOST_AUTO_TEST_CASE( MatrixCOOtoCSC_SLOW )
{
    SPM::Timing::CTimeKeeper tk;
    for( int i = 0; i < 20000000; i++ ) {
        SPM::Sparse::CMatrixCSC CSC;
        tk.StartTimer();
        SPM::Sparse::MatrixCOOtoCSC( A1_coo_expected_colwise, CSC, false );
        tk.EndTimer();
    }
    std::cout << "MatrixCOOtoCSC_SLOW tk.TimePerOp() = " << tk.TimePerOp() * 1.0e6 << " [us]" << std::endl;

    SPM::Sparse::CMatrixCSC CSC;
    SPM::Sparse::MatrixCOOtoCSC( A1_coo_expected, CSC, false );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_val.begin(), CSC.csc_val.end(), A1_csc_val_expected.begin(), A1_csc_val_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_kk.begin(), CSC.csc_kk.end(), A1_csc_kk_expected.begin(), A1_csc_kk_expected.end() );
    BOOST_CHECK_EQUAL_COLLECTIONS( CSC.csc_first.begin(), CSC.csc_first.end(), A1_csc_first_expected.begin(), A1_csc_first_expected.end() );
}
