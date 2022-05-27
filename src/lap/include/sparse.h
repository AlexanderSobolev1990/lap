//----------------------------------------------------------------------------------------------------------------------
///
/// \file       sparce.h
/// \brief      Работа с разреженными матрицами
/// \details    Перевод матрицы из плотного представления в COO, CSR, CSC вид
/// \date       27.05.22 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_SPARSE_H
#define SPML_SPARSE_H

// System includes:
#include <limits>
#include <armadillo>
#include <algorithm>

// SPML includes:
#include <compare.h>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace Sparse /// Разреженные матрицы
{
//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Структура хранения матрицы в COO формате (Coordinate list)
/// \details Матрица A[n,m] (n - число строк, m - число столбцов)
///
template<typename T>
struct CMatrixCOO
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    std::vector<T> Value;           ///< Вектор ненулевых элементов матрицы A[n,m] (n - число строк, m - число столбцов), размер nnz
    std::vector<unsigned> Row;      ///< Индексы строк ненулевых элементов
    std::vector<unsigned> Column;   ///< Индексы столбцов ненулевых элементов
};

///
/// \brief Структура хранения матрицы в CSR формате (Compressed Sparse Row Yale format)
/// \details Матрица A[n,m] (n - число строк, m - число столбцов)
///
template<typename T>
struct CMatrixCSR
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    std::vector<T> CSR;             ///< Вектор ненулевых элементов матрицы A[n,m] (n - число строк, m - число столбцов), размер nnz
    std::vector<unsigned> first;    ///< Вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
    std::vector<unsigned> kk;       ///< вектор начальных смещений в векторе CSR, размер n+1
};

///
/// \brief Структура хранения матрицы в CSC формате (Compressed Sparse Column Yale format)
/// \details Матрица A[n,m] (n - число строк, m - число столбцов)
///
template<typename T>
struct CMatrixCSC
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    std::vector<T> CSC;             ///< Вектор ненулевых элементов матрицы A[n,m] (n - число строк, m - число столбцов), размер nnz
    std::vector<unsigned> first;    ///< Вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
    std::vector<unsigned> kk;       ///< вектор начальных смещений в векторе CSC, размер m+1
};

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование плотной матрицы в COO формат (Coordinated list)
/// \param[in]  A   - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] COO - вектор ненулевых элементов матрицы A, размер nnz
/// \param[out] row - индексы строк ненулевых элементов, размер nnz
/// \param[out] col - индексы столбцов ненулевых элементов, размер nnz
///
void MatrixDenseToCOO( const arma::mat &A, std::vector<double> &COO, std::vector<unsigned> &row, std::vector<unsigned> &col );

///
/// \brief Преобразование матрицы из CSR формата (Compressed Sparse Row Yale format) в плотную матрицу
/// \param[in]  COO - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  row - индексы строк ненулевых элементов, размер nnz
/// \param[in]  col - индексы столбцов ненулевых элементов, размер nnz
/// \param[out] A   - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
///
void MatrixCOOtoDense( const std::vector<double> &COO, const std::vector<unsigned> &row, const std::vector<unsigned> &col, arma::mat &A );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование плотной матрицы в CSR формат (Compressed Sparse Row Yale format)
/// \details Данный способ хранения эффективен, если кол-во ненулевых элементов NNZ<(m*(n-1)-1)/2
/// \param[in]  A     - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] CSR   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[out] first - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[out] kk    - вектор начальных смещений в векторе CSR, размер n+1
///
void MatrixDenseToCSR( const arma::mat &A, std::vector<double> &CSR, std::vector<unsigned> &first, std::vector<unsigned> &kk );

///
/// \brief Преобразование матрицы из CSR формата (Compressed Sparse Row Yale format) в плотную матрицу
/// \param[in]  CSR   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  first - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in]  kk    - вектор начальных смещений в векторе CSR, размер n+1
/// \param[out] A     - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
///
void MatrixCSRtoDense( const std::vector<double> &CSR, const std::vector<unsigned> &first, const std::vector<unsigned> &kk, arma::mat &A );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование плотной матрицы в CSC формат (Compressed Sparse Column Yale format)
/// \details Данный способ хранения эффективен, если кол-во ненулевых элементов NNZ<(m*(n-1)-1)/2
/// \param[in]  A     - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] CSC   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[out] first - вектор индексов строк ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[out] kk    - вектор начальных смещений в векторе CSR, размер m+1
///
void MatrixDenseToCSC( const arma::mat &A, std::vector<double> &CSC, std::vector<unsigned> &first, std::vector<unsigned> &kk );

///
/// \brief Преобразование матрицы из CSV формата (Compressed Sparse Column Yale format) в плотную матрицу
/// \param[in]  CSR   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  first - вектор индексов строк ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in]  kk    - вектор начальных смещений в векторе CSR, размер m+1
/// \param[out] A     - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
///
void MatrixCSCtoDense( const std::vector<double> &CSC, const std::vector<unsigned> &first, const std::vector<unsigned> &kk, arma::mat &A );



} // end namespace Sparse
} // end namespace SPML
#endif // SPML_SPARSE_H
/// \}
