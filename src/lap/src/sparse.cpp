//----------------------------------------------------------------------------------------------------------------------
///
/// \file       sparse.cpp
/// \brief      Работа с разреженными матрицами
/// \details    Перевод матрицы из плотного представления в COO, CSR, CSC вид
/// \date       27.05.22 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#include <sparse.h>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace Sparse /// Решение задачи о назначениях
{
//----------------------------------------------------------------------------------------------------------------------

void MatrixDenseToCOO( const arma::mat &A, std::vector<double> &coo_val, std::vector<int> &coo_row,
    std::vector<int> &coo_col )
{
    coo_val.clear();
    coo_row.clear();
    coo_col.clear();

    int nnz = arma::accu( A != 0 ); // Число ненулевых элементов
    coo_val.reserve( nnz );
    coo_row.reserve( nnz );
    coo_col.reserve( nnz );

    for( unsigned long long i = 0; i < A.n_rows; i++ ) { // Cтроки
        for( unsigned long long j = 0; j < A.n_cols; j++ ) { // Cтолбцы
            if( !Compare::IsZeroAbs( A(i,j) ) ) {
                coo_val.push_back( A(i, j) );
                coo_row.push_back( i );
                coo_col.push_back( j );
            }
        }
    }
}

void MatrixDenseToCOO( const arma::mat &A, CMatrixCOO &COO )
{
    MatrixDenseToCOO( A, COO.coo_val, COO.coo_row, COO.coo_col );
}

void MatrixCOOtoDense( const std::vector<double> &coo_val, const std::vector<int> &coo_row,
    const std::vector<int> &coo_col, arma::mat &A )
{
    int n = *( std::max_element( coo_row.begin(), coo_row.end() ) ) + 1;
    int m = *( std::max_element( coo_col.begin(), coo_col.end() ) ) + 1;
    A = arma::mat( n, m, arma::fill::zeros );

    for( unsigned k = 0; k < coo_val.size(); k++ ) {
        A( coo_row[k], coo_col[k] ) = coo_val[k];
    }
}

void MatrixCOOtoDense( const CMatrixCOO &COO, arma::mat &A )
{
    MatrixCOOtoDense( COO.coo_val, COO.coo_row, COO.coo_col, A );
}

//----------------------------------------------------------------------------------------------------------------------

void MatrixDenseToCSR( const arma::mat &A, std::vector<double> &csr_val, std::vector<int> &csr_kk,
    std::vector<int> &csr_first )
{
    csr_val.clear();
    csr_kk.clear();
    csr_first.clear();

    int nnz = arma::accu( A != 0 ); // Число ненулевых элементов
    csr_val.reserve( nnz );
    csr_kk.reserve( nnz );
    csr_first.reserve( A.n_rows + 1 );

    int nnz_in_row = 0; // Кол-во ненулевых элементов (non-zero) в строке
    csr_first.push_back(0);// Первый элемент надо занулить

    for( unsigned long long i = 0; i < A.n_rows; i++ ) { // Cтроки
        nnz_in_row = 0;
        for( unsigned long long j = 0; j < A.n_cols; j++ ) { // Cтолбцы
            if( !Compare::IsZeroAbs( A(i,j) ) ) { // if( A[i,j] != 0 )
                csr_val.push_back( A(i,j) );//CSR[nnz] = A(i,j);
                csr_kk.push_back( j );
                nnz_in_row++;
            }
        }
        csr_first.push_back( csr_first[i] + nnz_in_row );
    }    
}

void MatrixDenseToCSR( const arma::mat &A, CMatrixCSR &CSR )
{
    MatrixDenseToCSR( A, CSR.csr_val, CSR.csr_kk, CSR.csr_first );
}

void MatrixCSRtoDense( const std::vector<double> &csr_val, const std::vector<int> &csr_kk,
    const std::vector<int> &csr_first, arma::mat &A )
{
    int n = csr_first.size() - 1;
    int m = *( std::max_element( csr_kk.begin(), csr_kk.end() ) ) + 1;
    A = arma::mat( n, m, arma::fill::zeros );

    for( int i = 0; i < n; i++ ) {
        int nnz_row = csr_first[i];
        int nnz_row_next = csr_first[i+1];
        for( int j = nnz_row; j < nnz_row_next; j++ ) {
            A(i, csr_kk[j]) = csr_val[j];//C[m*i+kk[j]] = CSR[j];
        }
    }
}

void MatrixCSRtoDense( const CMatrixCSR &CSR, arma::mat &A )
{
    MatrixCSRtoDense( CSR.csr_val, CSR.csr_kk, CSR.csr_first, A );
}

//----------------------------------------------------------------------------------------------------------------------

void MatrixDenseToCSC( const arma::mat &A, std::vector<double> &csc_val, std::vector<int> &csc_kk,
    std::vector<int> &csc_first )
{
    csc_val.clear();
    csc_kk.clear();
    csc_first.clear();

    int nnz = arma::accu( A != 0 ); // Число ненулевых элементов
    csc_val.reserve( nnz );
    csc_kk.reserve( nnz );
    csc_first.reserve( A.n_cols + 1 );

    int nnz_in_col = 0; // Кол-во ненулевых элементов (non-zero) в столбце
    csc_first.push_back(0);// Первый элемент надо занулить

    for( unsigned long long j = 0; j < A.n_cols; j++ ) { // Cтолбцы
        nnz_in_col = 0;
        for( unsigned long long i = 0; i < A.n_rows; i++ ) { // Cтроки
            if( !Compare::IsZeroAbs( A(i,j) ) ) { // if( A[i,j] != 0 )
                csc_val.push_back( A(i,j) );//CSC[nnz] = A(i,j);
                csc_kk.push_back( i );
                nnz_in_col++;
            }
        }
        csc_first.push_back( csc_first[j] + nnz_in_col );
    }
}

void MatrixDenseToCSC( const arma::mat &A, CMatrixCSC &CSC )
{
    MatrixDenseToCSC( A, CSC.csc_val, CSC.csc_kk, CSC.csc_first );
}

void MatrixCSCtoDense( const std::vector<double> &csc_val, const std::vector<int> &csc_kk,
    const std::vector<int> &csc_first, arma::mat &A )
{
    int n = *( std::max_element( csc_kk.begin(), csc_kk.end() ) ) + 1;
    int m = csc_first.size() - 1;
    A = arma::mat( n, m, arma::fill::zeros );

    for( int j = 0; j < m; j++ ) {
        int nnz_col = csc_first[j];
        int nnz_col_next = csc_first[j+1];
        for( int i = nnz_col; i < nnz_col_next; i++ ) {
            A(csc_kk[i], j) = csc_val[i];
        }
    }
}

void MatrixCSCtoDense( const CMatrixCSC &CSC, arma::mat &A )
{
    MatrixCSCtoDense( CSC.csc_val, CSC.csc_kk, CSC.csc_first, A );
}

//----------------------------------------------------------------------------------------------------------------------

void MatrixCOOtoCSR( const std::vector<double> &coo_val, const std::vector<int> &coo_row,
    const std::vector<int> &coo_col, std::vector<double> &csr_val, std::vector<int> &csr_kk,
    std::vector<int> &csr_first, bool sorted )
{
    csr_val.clear();
    csr_kk.clear();
    csr_first.clear();
    int nnz = coo_val.size(); // Число ненулевых элементов
    int n = *( std::max_element( coo_row.begin(), coo_row.end() ) ) + 1; // n_rows

    if( sorted ) { // https://stackoverflow.com/questions/23583975/convert-coo-to-csr-format-in-c
        csr_val = coo_val;
        csr_kk = coo_col;
        for( int i = 0; i <= n; i++ ) {
            csr_first.push_back( 0 );
        }
        for( int i = 0; i < nnz; i++ ) {
            csr_first[coo_row[i] + 1]++;
        }
        for( int i = 0; i < n; i++ ) {
            csr_first[i + 1] += csr_first[i];
        }
    } else { // https://github.com/scipy/scipy/blob/3b36a57/scipy/sparse/sparsetools/coo.h#L34
        for( int i = 0; i < nnz; i++ ) {
            csr_val.push_back( 0 );
            csr_kk.push_back( 0 );
        }
        for( int i = 0; i <= n; i++ ) {
            csr_first.push_back( 0 );
        }

        // Bp - kk
        // Bi - first
        // Bx - CSR

        // Ax - COO
        // Ai - row
        // Aj - col

        //compute number of non-zero entries per row of A
        for( int k = 0; k < nnz; k++ ) {
            csr_first[coo_row[k]]++;
        }

        //cumsum the nnz per row to get Bp[]
        for( int i = 0, cumsum = 0; i < n; i++ ) {
            int temp = csr_first[i];
            csr_first[i] = cumsum;
            cumsum += temp;
        }
        csr_first[n] = nnz;

        //write Aj,Ax into Bj,Bx
        for( int k = 0; k < nnz; k++ ) {
            int row_ = coo_row[k];
            int dest = csr_first[row_];

            csr_kk[dest] = coo_col[k];
            csr_val[dest] = coo_val[k];

            csr_first[row_]++;
        }

        for( int i = 0, last = 0; i <= n; i++ ) {
            int temp = csr_first[i];
            csr_first[i] = last;
            last = temp;
        }
        //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
    }
}

void MatrixCOOtoCSR( const CMatrixCOO &COO, CMatrixCSR &CSR, bool sorted )
{
    MatrixCOOtoCSR( COO.coo_val, COO.coo_row, COO.coo_col, CSR.csr_val, CSR.csr_kk, CSR.csr_first, sorted );
}

void MatrixCOOtoCSC( const std::vector<double> &coo_val, const std::vector<int> &coo_row,
    const std::vector<int> &coo_col, std::vector<double> &csc_val, std::vector<int> &csc_kk,
    std::vector<int> &csc_first, bool sorted )
{  
    csc_val.clear();
    csc_kk.clear();
    csc_first.clear();
    int nnz = coo_val.size(); // Число ненулевых элементов
    int m = *( std::max_element( coo_col.begin(), coo_col.end() ) ) + 1; // m_cols

    if( sorted ) { // https://stackoverflow.com/questions/23583975/convert-coo-to-csr-format-in-c
        csc_val = coo_val;
        csc_kk = coo_row;
        for( int i = 0; i <= m; i++ ) {
            csc_first.push_back( 0 );
        }
        for( int i = 0; i < nnz; i++ ) {
            csc_first[coo_col[i] + 1]++;
        }
        for( int i = 0; i < m; i++ ) {
            csc_first[i + 1] += csc_first[i];
        }
    } else { // https://github.com/scipy/scipy/blob/3b36a57/scipy/sparse/sparsetools/coo.h#L34
        for( int i = 0; i < nnz; i++ ) {
            csc_val.push_back( 0 );
            csc_kk.push_back( 0 );
        }
        for( int i = 0; i <= m; i++ ) {
            csc_first.push_back( 0 );
        }

        // Bp - kk
        // Bi - first
        // Bx - CSR

        // Ax - COO
        // Ai - row
        // Aj - col

        //compute number of non-zero entries per col of A
        for( int k = 0; k < nnz; k++ ) {
            csc_first[coo_col[k]]++;
        }

        //cumsum the nnz per col to get Bp[]
        for( int j = 0, cumsum = 0; j < m; j++ ) {
            int temp = csc_first[j];
            csc_first[j] = cumsum;
            cumsum += temp;
        }
        csc_first[m] = nnz;

        //write Aj,Ax into Bj,Bx
        for(int k = 0; k < nnz; k++) {
            int col_ = coo_col[k];
            int dest = csc_first[col_];

            csc_kk[dest] = coo_row[k];
            csc_val[dest] = coo_val[k];

            csc_first[col_]++;
        }

        for( int j = 0, last = 0; j <= m; j++ ) {
            int temp = csc_first[j];
            csc_first[j] = last;
            last = temp;
        }
        //now Bp,Bj,Bx form a CSC representation (with possible duplicates)
    }
}

void MatrixCOOtoCSC( const CMatrixCOO &COO, CMatrixCSC &CSC, bool sorted )
{
    MatrixCOOtoCSC( COO.coo_val, COO.coo_row, COO.coo_col, CSC.csc_val, CSC.csc_kk, CSC.csc_first, sorted );
}

} // end namespace Sparse
} // end namespace SPML
/// \}

