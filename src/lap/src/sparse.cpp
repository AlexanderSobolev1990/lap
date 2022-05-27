//----------------------------------------------------------------------------------------------------------------------
///
/// \file       sparce.cpp
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

void MatrixDenseToCOO( const arma::mat &A, std::vector<double> &COO, std::vector<unsigned> &row, std::vector<unsigned> &col )
{
    COO.clear();
    row.clear();
    col.clear();

    int nnz = arma::accu( A != 0 ); // Число ненулевых элементов
    COO.reserve( nnz );
    row.reserve( nnz );
    col.reserve( nnz );

    for( unsigned long long i = 0; i < A.n_rows; i++ ) { // Cтроки
        for( unsigned long long j = 0; j < A.n_cols; j++ ) { // Cтолбцы
            if( !Compare::IsZeroAbs( A(i,j) ) ) {
                COO.push_back( A(i, j) );
                row.push_back( i );
                col.push_back( j );
            }
        }
    }
}

void MatrixCOOtoDense( const std::vector<double> &COO, const std::vector<unsigned> &row, const std::vector<unsigned> &col, arma::mat &A )
{
    int n = *( std::max_element( row.begin(), row.end() ) ) + 1;
    int m = *( std::max_element( col.begin(), col.end() ) ) + 1;
    A = arma::mat( n, m, arma::fill::zeros );

    for( unsigned k = 0; k < COO.size(); k++ ) {
        A( row[k], col[k] ) = COO[k];
    }
}

//----------------------------------------------------------------------------------------------------------------------

void MatrixDenseToCSR( const arma::mat &A, std::vector<double> &CSR, std::vector<unsigned> &first, std::vector<unsigned> &kk )
{
    CSR.clear();
    first.clear();
    kk.clear();

    int nnz = arma::accu( A != 0 ); // Число ненулевых элементов
    CSR.reserve( nnz );
    first.reserve( nnz );
    kk.reserve( A.n_rows + 1 );

    int nnz_in_row = 0; // Кол-во ненулевых элементов (non-zero) в строке
    kk.push_back(0);// kk[0] = 0; // Первый элемент надо занулить

    for( unsigned long long i = 0; i < A.n_rows; i++ ) { // Cтроки
        nnz_in_row = 0;
        for( unsigned long long j = 0; j < A.n_cols; j++ ) { // Cтолбцы
            if( !Compare::IsZeroAbs( A(i,j) ) ) { // if( C[i,j] != 0 )
                CSR.push_back( A(i,j) );//CSR[nnz] = A(i,j);
                first.push_back( j );//first[nnz] = j;
//                nnz++;
                nnz_in_row++;
            }
        }
//        kk[i+1] = kk[i] + nnz_in_row;
        kk.push_back( kk[i] + nnz_in_row );
    }    
}

void MatrixCSRtoDense( const std::vector<double> &CSR, const std::vector<unsigned> &first, const std::vector<unsigned> &kk, arma::mat &A )
{
    int n = kk.size() - 1;
    int m = *( std::max_element( first.begin(), first.end() ) ) + 1;
    A = arma::mat( n, m, arma::fill::zeros );

    for( int i = 0; i < n; i++ ) {
        int nnz_row = kk[i];
        int nnz_row_next = kk[i+1];
        for( int j = nnz_row; j < nnz_row_next; j++ ) {
            A(i, first[j]) = CSR[j];//C[m*i+first[j]] = CSR[j];
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void MatrixDenseToCSC( const arma::mat &A, std::vector<double> &CSC, std::vector<unsigned> &first, std::vector<unsigned> &kk )
{
    CSC.clear();
    first.clear();
    kk.clear();

    int nnz = arma::accu( A != 0 ); // Число ненулевых элементов
    CSC.reserve( nnz );
    first.reserve( nnz );
    kk.reserve( A.n_cols + 1 );

    int nnz_in_col = 0; // Кол-во ненулевых элементов (non-zero) в столбце
    kk.push_back(0);// kk[0] = 0; // Первый элемент надо занулить

    for( unsigned long long j = 0; j < A.n_cols; j++ ) { // Cтолбцы
        nnz_in_col = 0;
        for( unsigned long long i = 0; i < A.n_rows; i++ ) { // Cтроки
            if( !Compare::IsZeroAbs( A(i,j) ) ) { // if( C[i,j] != 0 )
                CSC.push_back( A(i,j) );//CSR[nnz] = A(i,j);
                first.push_back( i );//first[nnz] = i;
//                nnz++;
                nnz_in_col++;
            }
        }
//        kk[i+1] = kk[j] + nnz_in_row;
        kk.push_back( kk[j] + nnz_in_col );
    }
}

void MatrixCSCtoDense( const std::vector<double> &CSC, const std::vector<unsigned> &first, const std::vector<unsigned> &kk, arma::mat &A )
{
    int n = *( std::max_element( first.begin(), first.end() ) ) + 1;
    int m = kk.size() - 1;
    A = arma::mat( n, m, arma::fill::zeros );

    for( int j = 0; j < m; j++ ) {
        int nnz_col = kk[j];
        int nnz_col_next = kk[j+1];
        for( int i = nnz_col; i < nnz_col_next; i++ ) {
            A(first[i], j) = CSC[i];
        }
    }
}

} // end namespace Sparse
} // end namespace SPML
/// \}

