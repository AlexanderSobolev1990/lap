//----------------------------------------------------------------------------------------------------------------------
///
/// \file       sparse.hpp
/// \brief      Работа с разреженными матрицами
/// \details    Перевод матрицы из плотного представления в COO (Coordinate list), CSR (Compressed Sparse Row Yale
/// format), CSC (Compressed Sparse Column Yale format) вид.
/// \date       27.05.22 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_SPARSE_HPP_
#define SPML_SPARSE_HPP_

// System includes:
#include <limits>
#include <armadillo>
#include <algorithm>

// SPML includes:
#include <compare.hpp>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace Sparse /// Разреженные матрицы
{
//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Структура хранения матрицы в координатном COO формате (Coordinate list)
/// \details Матрица A[n,m] (n - число строк, m - число столбцов)
///
struct CMatrixCOO
{
    std::vector<double> coo_val; ///< Вектор ненулевых элементов матрицы A[n,m] (n - число строк, m - число столбцов), размер nnz
    std::vector<int> coo_row; ///< Индексы строк ненулевых элементов
    std::vector<int> coo_col; ///< Индексы столбцов ненулевых элементов

    ///
    /// \brief Число строк
    ///
    inline int n_rows() const
    {
        return coo_row.empty() ? 0 : *std::max_element( coo_row.begin(), coo_row.end() ) + 1; 
    }

    ///
    /// \brief Число столбцов
    ///
    inline int n_cols() const
    {
        return coo_col.empty() ? 0 : *std::max_element( coo_col.begin(), coo_col.end() ) + 1; 
    }
};

///
/// \brief Структура хранения матрицы в CSR формате (построчно) (Compressed Sparse Row Yale format)
/// \details Матрица A[n,m] (n - число строк, m - число столбцов)
///
struct CMatrixCSR
{
    std::vector<double> csr_val; ///< Вектор ненулевых элементов матрицы A[n,m] (n - число строк, m - число столбцов), размер nnz
    std::vector<int> csr_kk; ///< Вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
    std::vector<int> csr_first; ///< Вектор начальных смещений в векторе CSR, размер n+1

    ///
    /// \brief Число строк
    ///
    inline int n_rows() const
    {
        return static_cast<int>( csr_first.size() ) - 1;
    }

    ///
    /// \brief Число столбцов
    ///
    inline int n_cols() const
    {
        int max_kk = -1;
        for( unsigned i = 0; i < csr_kk.size(); i++ ) { // Поиск максимального элемента в массиве kk - это и будет кол-во столбцов
            if( csr_kk[i] > max_kk ) {
                max_kk = csr_kk[i];
            }
        }
        return ( max_kk + 1 );
    }
};

///
/// \brief Структура хранения матрицы в CSC формате (по столбцам) (Compressed Sparse Column Yale format)
/// \details Матрица A[n,m] (n - число строк, m - число столбцов)
///
struct CMatrixCSC
{
    std::vector<double> csc_val; ///< Вектор ненулевых элементов матрицы A[n,m] (n - число строк, m - число столбцов), размер nnz
    std::vector<int> csc_kk; ///< Вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
    std::vector<int> csc_first; ///< вектор начальных смещений в векторе CSC, размер m+1

    ///
    /// \brief Число строк
    ///
    inline int n_rows() const
    {
        int max_kk = -1;
        for( unsigned i = 0; i < csc_kk.size(); i++ ) { // Поиск максимального элемента в массиве kk
            if( csc_kk[i] > max_kk ) {
                max_kk = csc_kk[i];
            }
        }
        return ( max_kk + 1 ); // Возвращаем количество строк (max_kk + 1 для нумерации с нуля)
    }

    ///
    /// \brief Число столбцов
    ///
    inline int n_cols() const
    {
        return static_cast<int>( csc_first.size() ) - 1; // Число столбцов равно размеру csc_first - 1
    }
};

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование плотной матрицы в COO формат (Coordinated list)
/// \param[in] A - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] coo_val - вектор ненулевых элементов матрицы A, размер nnz
/// \param[out] coo_row - индексы строк ненулевых элементов, размер nnz
/// \param[out] coo_col - индексы столбцов ненулевых элементов, размер nnz
/// \param[in] empty - значение "нулевых" элементов
/// \param[in] abs_eps - абсолютная точность при сравнении действительных чисел
///
void MatrixDenseToCOO( const arma::mat &A, std::vector<double> &coo_val, std::vector<int> &coo_row,
    std::vector<int> &coo_col, double empty = 0.0, double abs_eps = 1.0e-6 );

///
/// \brief Преобразование плотной матрицы в COO формат (Coordinated list)
/// \param[in] A - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] COO - cтруктура хранения матрицы в COO формате (Coordinate list)
/// \param[in] empty - значение "нулевых" элементов
/// \param[in] abs_eps - абсолютная точность при сравнении действительных чисел
///
void MatrixDenseToCOO( const arma::mat &A, CMatrixCOO &COO, double empty = 0.0, double abs_eps = 1.0e-6 );

///
/// \brief Преобразование матрицы из COO формата (Coordinated list) в плотную матрицу
/// \param[in] coo_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in] coo_row - индексы строк ненулевых элементов, размер nnz
/// \param[in] coo_col - индексы столбцов ненулевых элементов, размер nnz
/// \param[out] A - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[in] empty - значение "нулевых" элементов
///
void MatrixCOOtoDense( const std::vector<double> &coo_val, const std::vector<int> &coo_row,
    const std::vector<int> &coo_col, arma::mat &A, double empty = 0.0 );

///
/// \brief Преобразование матрицы из COO формата (Coordinated list) в плотную матрицу
/// \param[in] COO - cтруктура хранения матрицы в COO формате (Coordinate list)
/// \param[out] A - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[in] empty - значение "нулевых" элементов
///
void MatrixCOOtoDense( const CMatrixCOO &COO, arma::mat &A, double empty = 0.0 );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование плотной матрицы в CSR формат (Compressed Sparse Row Yale format)
/// \details Данный способ хранения эффективен, если кол-во ненулевых элементов NNZ<(m*(n-1)-1)/2
/// \param[in] A - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] csr_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[out] csr_kk - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[out] csr_first - вектор начальных смещений в векторе CSR, размер n+1
/// \param[in] empty - значение "нулевых" элементов
/// \param[in] abs_eps - абсолютная точность при сравнении действительных чисел
///
void MatrixDenseToCSR( const arma::mat &A, std::vector<double> &csr_val, std::vector<int> &csr_kk,
    std::vector<int> &csr_first, double empty = 0.0, double abs_eps = 1.0e-6 );

///
/// \brief Преобразование плотной матрицы в CSR формат (Compressed Sparse Row Yale format)
/// \param[in] A - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] CSR - cтруктура хранения матрицы в CSR формате (Compressed Sparse Row Yale format)
/// \param[in] empty - значение "нулевых" элементов
/// \param[in] abs_eps - абсолютная точность при сравнении действительных чисел
///
void MatrixDenseToCSR( const arma::mat &A, CMatrixCSR &CSR, double empty = 0.0, double abs_eps = 1.0e-6 );

///
/// \brief Преобразование матрицы из CSR формата (Compressed Sparse Row Yale format) в плотную матрицу
/// \param[in] csr_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in] csr_kk - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in] csr_first - вектор начальных смещений в векторе CSR, размер n+1
/// \param[out] A - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[in] empty - значение "нулевых" элементов
///
void MatrixCSRtoDense( const std::vector<double> &csr_val, const std::vector<int> &csr_kk,
    const std::vector<int> &csr_first, arma::mat &A, double empty = 0.0 );

///
/// \brief Преобразование матрицы из CSR формата (Compressed Sparse Row Yale format) в плотную матрицу
/// \param[in] CSR - cтруктура хранения матрицы в CSR формате (Compressed Sparse Row Yale format)
/// \param[out] A - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[in] empty - значение "нулевых" элементов
///
void MatrixCSRtoDense( const CMatrixCSR &CSR, arma::mat &A, double empty = 0.0 );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование плотной матрицы в CSC формат (Compressed Sparse Column Yale format)
/// \details Данный способ хранения эффективен, если кол-во ненулевых элементов NNZ<(m*(n-1)-1)/2
/// \param[in] A - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] csc_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[out] csc_kk - вектор индексов строк ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[out] csc_first - вектор начальных смещений в векторе CSR, размер m+1
/// \param[in] empty - значение "нулевых" элементов
/// \param[in] abs_eps - абсолютная точность при сравнении действительных чисел
///
void MatrixDenseToCSC( const arma::mat &A, std::vector<double> &csc_val, std::vector<int> &csc_kk,
    std::vector<int> &csc_first, double empty = 0.0, double abs_eps = 1.0e-6 );

///
/// \brief Преобразование плотной матрицы в CSC формат (Compressed Sparse Column Yale format)
/// \param[in] A - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] CSC - cтруктура хранения матрицы в CSC формате (Compressed Sparse Column Yale format)
/// \param[in] empty - значение "нулевых" элементов
/// \param[in] abs_eps - абсолютная точность при сравнении действительных чисел
///
void MatrixDenseToCSC( const arma::mat &A, CMatrixCSC &CSC, double empty = 0.0, double abs_eps = 1.0e-6 );

///
/// \brief Преобразование матрицы из CSC формата (Compressed Sparse Column Yale format) в плотную матрицу
/// \param[in] csc_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in] csc_kk - вектор индексов строк ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in] csc_first - вектор начальных смещений в векторе CSR, размер m+1
/// \param[out] A - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[in] empty - значение "нулевых" элементов
///
void MatrixCSCtoDense( const std::vector<double> &csc_val, const std::vector<int> &csc_kk,
    const std::vector<int> &csc_first, arma::mat &A, double empty = 0.0 );

///
/// \brief Преобразование матрицы из CSC формата (Compressed Sparse Column Yale format) в плотную матрицу
/// \param[in] CSC - cтруктура хранения матрицы в CSC формате (Compressed Sparse Column Yale format)
/// \param[out] A - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[in] empty - значение "нулевых" элементов
///
void MatrixCSCtoDense( const CMatrixCSC &CSC, arma::mat &A, double empty = 0.0 );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование матрицы в COO формате (Coordinate list) в CSR формат (Compressed Sparse Row Yale format)
/// \param[in] coo_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in] coo_row - индексы строк ненулевых элементов, размер nnz
/// \param[in] coo_col - индексы столбцов ненулевых элементов, размер nnz
/// \param[in] csr_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in] csr_kk - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in] csr_first - вектор начальных смещений в векторе CSR, размер n+1
/// \param[in] sorted - признак отсортированности COO входа по строкам (даёт ускорение в ~2 раза)
///
void MatrixCOOtoCSR( const std::vector<double> &coo_val, const std::vector<int> &coo_row,
    const std::vector<int> &coo_col, std::vector<double> &csr_val, std::vector<int> &csr_kk,
    std::vector<int> &csr_first, bool sorted = false );

///
/// \brief Преобразование матрицы в COO формате (Coordinate list) в CSR формат (Compressed Sparse Row Yale format)
/// \param[in] COO - матрица в COO формате (Coordinate list)
/// \param[out] CSR - матрица в CSR формате (Compressed Sparse Row Yale format)
/// \param[in] sorted - признак отсортированности COO входа по строкам (даёт ускорение в ~2 раза)
///
void MatrixCOOtoCSR( const CMatrixCOO &COO, CMatrixCSR &CSR, bool sorted = false );

///
/// \brief Преобразование матрицы в COO формате (Coordinate list) в CSC формат (Compressed Sparse Column Yale format)
/// \param[in] coo_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in] coo_row - индексы строк ненулевых элементов, размер nnz
/// \param[in] coo_col - индексы столбцов ненулевых элементов, размер nnz
/// \param[in] csc_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in] csc_kk - вектор индексов строк ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in] csc_first - вектор начальных смещений в векторе CSR, размер m+1
/// \param[in] sorted - признак отсортированности COO входа по столбцам (даёт ускорение в ~2 раза)
///
void MatrixCOOtoCSC( const std::vector<double> &coo_val, const std::vector<int> &coo_row,
    const std::vector<int> &coo_col, std::vector<double> &csc_val, std::vector<int> &csc_kk,
    std::vector<int> &csc_first, bool sorted = false );

///
/// \brief Преобразование матрицы в COO формате (Coordinate list) в CSC формат (Compressed Sparse Column Yale format)
/// \param[in] COO - матрица в COO формате (Coordinate list)
/// \param[out] CSC - матрица в CSR формате (Compressed Sparse Column Yale format)
/// \param[in] sorted - признак отсортированности COO входа по столбцам (даёт ускорение в ~2 раза)
///
void MatrixCOOtoCSC( const CMatrixCOO &COO, CMatrixCSC &CSC, bool sorted = false );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Ключ элемента Aij матрицы A в COO формате (Coordinate list)
/// \attention Оператор < перегружен для случая построчного хранения
///
struct CKeyCOO
{
public:
    ///
    /// \brief i - индекс строки
    ///
    int i() const { return i_; }

    ///
    /// \brief j - индекс столбца
    ///
    int j() const { return j_; }

    ///
    /// \brief Конструктор по умолчанию
    ///
    CKeyCOO() : i_( 0 ), j_( 0 ){}

    ///
    /// \brief Параметрический конструктор
    /// \param[in] i - индекс строки
    /// \param[in] j - индекс столбца
    ///
    CKeyCOO( int i, int j ) : i_( i ), j_( j ){}

    bool operator <( CKeyCOO const& other ) const
    {
        if( ( this->i_ < other.i_ ) || ( ( this->i_ == other.i_ ) && ( this->j_ < other.j_ ) ) ) {
            return true;
        }
        return false;
    }

private:
    int i_; ///< Индекс строки
    int j_; ///< Индекс столбца
};

} // end namespace Sparse
} // end namespace SPML
#endif
/// \}
