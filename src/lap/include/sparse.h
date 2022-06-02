//----------------------------------------------------------------------------------------------------------------------
///
/// \file       sparse.h
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
    std::vector<T> coo_val;         ///< Вектор ненулевых элементов матрицы A[n,m] (n - число строк, m - число столбцов), размер nnz
//    std::vector<unsigned> coo_row;  ///< Индексы строк ненулевых элементов
//    std::vector<unsigned> coo_col;  ///< Индексы столбцов ненулевых элементов
    std::vector<int> coo_row;  ///< Индексы строк ненулевых элементов
    std::vector<int> coo_col;  ///< Индексы столбцов ненулевых элементов
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
    std::vector<T> csr_val;             ///< Вектор ненулевых элементов матрицы A[n,m] (n - число строк, m - число столбцов), размер nnz
//    std::vector<unsigned> csr_first;    ///< Вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
//    std::vector<unsigned> csr_kk;       ///< вектор начальных смещений в векторе CSR, размер n+1
    std::vector<int> csr_first;    ///< Вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
    std::vector<int> csr_kk;       ///< вектор начальных смещений в векторе CSR, размер n+1
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
    std::vector<T> csc_val;             ///< Вектор ненулевых элементов матрицы A[n,m] (n - число строк, m - число столбцов), размер nnz
//    std::vector<unsigned> csc_first;    ///< Вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
//    std::vector<unsigned> csc_kk;       ///< вектор начальных смещений в векторе CSC, размер m+1
    std::vector<int> csc_first;    ///< Вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
    std::vector<int> csc_kk;       ///< вектор начальных смещений в векторе CSC, размер m+1
};

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование плотной матрицы в COO формат (Coordinated list)
/// \param[in]  A       - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] coo_val - вектор ненулевых элементов матрицы A, размер nnz
/// \param[out] coo_row - индексы строк ненулевых элементов, размер nnz
/// \param[out] coo_col - индексы столбцов ненулевых элементов, размер nnz
///
void MatrixDenseToCOO( const arma::mat &A, std::vector<double> &coo_val, std::vector<int> &coo_row,
    std::vector<int> &coo_col );
//void MatrixDenseToCOO( const arma::mat &A, std::vector<double> &coo_val, std::vector<unsigned> &coo_row,
//    std::vector<unsigned> &coo_col );

///
/// \brief Преобразование плотной матрицы в COO формат (Coordinated list)
/// \param[in]  A   - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] COO - cтруктура хранения матрицы в COO формате (Coordinate list)
///
template<typename T>
void MatrixDenseToCOO( const arma::mat &A, CMatrixCOO<T> &COO )
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    MatrixDenseToCOO( A, COO.coo_val, COO.coo_row, COO.coo_col );
}

///
/// \brief Преобразование матрицы из COO формата (Coordinated list) в плотную матрицу
/// \param[in]  coo_val - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  coo_row - индексы строк ненулевых элементов, размер nnz
/// \param[in]  coo_col - индексы столбцов ненулевых элементов, размер nnz
/// \param[out] A       - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
///
void MatrixCOOtoDense( const std::vector<double> &coo_val, const std::vector<int> &coo_row,
    const std::vector<int> &coo_col, arma::mat &A );
//void MatrixCOOtoDense( const std::vector<double> &coo_val, const std::vector<unsigned> &coo_row,
//    const std::vector<unsigned> &coo_col, arma::mat &A );

///
/// \brief Преобразование матрицы из COO формата (Coordinated list) в плотную матрицу
/// \param[in]  COO - cтруктура хранения матрицы в COO формате (Coordinate list)
/// \param[out] A   - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
///
template<typename T>
void MatrixCOOtoDense( const CMatrixCOO<T> &COO, arma::mat &A )
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    MatrixCOOtoDense( COO.coo_val, COO.coo_row, COO.coo_col, A );
}

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование плотной матрицы в CSR формат (Compressed Sparse Row Yale format)
/// \details Данный способ хранения эффективен, если кол-во ненулевых элементов NNZ<(m*(n-1)-1)/2
/// \param[in]  A         - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] csr_val   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[out] csr_first - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[out] csr_kk    - вектор начальных смещений в векторе CSR, размер n+1
///
void MatrixDenseToCSR( const arma::mat &A, std::vector<double> &csr_val, std::vector<int> &csr_first,
    std::vector<int> &csr_kk );
//void MatrixDenseToCSR( const arma::mat &A, std::vector<double> &csr_val, std::vector<unsigned> &csr_first,
//    std::vector<unsigned> &csr_kk );

///
/// \brief Преобразование плотной матрицы в CSR формат (Compressed Sparse Row Yale format)
/// \param[in]  A   - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] CSR - cтруктура хранения матрицы в CSR формате (Compressed Sparse Row Yale format)
///
template<typename T>
void MatrixDenseToCSR( const arma::mat &A, CMatrixCSR<T> &CSR )
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    MatrixDenseToCSR( A, CSR.csr_val, CSR.csr_first, CSR.csr_kk );
}

///
/// \brief Преобразование матрицы из CSR формата (Compressed Sparse Row Yale format) в плотную матрицу
/// \param[in]  csr_val   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  csr_first - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in]  csr_kk    - вектор начальных смещений в векторе CSR, размер n+1
/// \param[out] A         - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
///
void MatrixCSRtoDense( const std::vector<double> &csr_val, const std::vector<int> &csr_first,
    const std::vector<int> &csr_kk, arma::mat &A );
//void MatrixCSRtoDense( const std::vector<double> &csr_val, const std::vector<unsigned> &csr_first,
//    const std::vector<unsigned> &csr_kk, arma::mat &A );

///
/// \brief Преобразование матрицы из CSR формата (Compressed Sparse Row Yale format) в плотную матрицу
/// \param[in]  CSR - cтруктура хранения матрицы в CSR формате (Compressed Sparse Row Yale format)
/// \param[out] A   - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
///
template<typename T>
void MatrixCSRtoDense( const CMatrixCSR<T> &CSR, arma::mat &A )
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    MatrixCSRtoDense( CSR.csr_val, CSR.csr_first, CSR.csr_kk, A );
}

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование плотной матрицы в CSC формат (Compressed Sparse Column Yale format)
/// \details Данный способ хранения эффективен, если кол-во ненулевых элементов NNZ<(m*(n-1)-1)/2
/// \param[in]  A         - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] csc_val   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[out] csc_first - вектор индексов строк ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[out] csc_kk    - вектор начальных смещений в векторе CSR, размер m+1
///
void MatrixDenseToCSC( const arma::mat &A, std::vector<double> &csc_val, std::vector<int> &csc_first,
    std::vector<int> &csc_kk );
//void MatrixDenseToCSC( const arma::mat &A, std::vector<double> &csc_val, std::vector<unsigned> &csc_first,
//    std::vector<unsigned> &csc_kk );

///
/// \brief Преобразование плотной матрицы в CSC формат (Compressed Sparse Column Yale format)
/// \param[in]  A   - исходная матрица, размер [n,m] (n - число строк, m - число столбцов)
/// \param[out] CSC - cтруктура хранения матрицы в CSC формате (Compressed Sparse Column Yale format)
///
template<typename T>
void MatrixDenseToCSC( const arma::mat &A, CMatrixCSC<T> &CSC )
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    MatrixDenseToCSC( A, CSC.csc_val, CSC.csc_first, CSC.csc_kk );
}

///
/// \brief Преобразование матрицы из CSC формата (Compressed Sparse Column Yale format) в плотную матрицу
/// \param[in]  csc_val   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  csc_first - вектор индексов строк ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in]  csc_kk    - вектор начальных смещений в векторе CSR, размер m+1
/// \param[out] A         - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
///
void MatrixCSCtoDense( const std::vector<double> &csc_val, const std::vector<int> &csc_first,
    const std::vector<int> &csc_kk, arma::mat &A );
//void MatrixCSCtoDense( const std::vector<double> &csc_val, const std::vector<unsigned> &csc_first,
//    const std::vector<unsigned> &csc_kk, arma::mat &A );

///
/// \brief Преобразование матрицы из CSC формата (Compressed Sparse Column Yale format) в плотную матрицу
/// \param[in]  CSC - cтруктура хранения матрицы в CSC формате (Compressed Sparse Column Yale format)
/// \param[out] A   - плотная матрица, размер [n,m] (n - число строк, m - число столбцов)
///
template<typename T>
void MatrixCSCtoDense( const CMatrixCSC<T> &CSC, arma::mat &A )
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    MatrixCSCtoDense( CSC.csc_val, CSC.csc_first, CSC.csc_kk, A );
}

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Преобразование матрицы в COO формате (Coordinate list) в CSR формат (Compressed Sparse Row Yale format)
/// \param[in]  coo_val   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  coo_row   - индексы строк ненулевых элементов, размер nnz
/// \param[in]  coo_col   - индексы столбцов ненулевых элементов, размер nnz
/// \param[in]  csr_val   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  csr_first - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in]  csr_kk    - вектор начальных смещений в векторе CSR, размер n+1
/// \param[in]  sorted    - признак отсортированности COO входа по строкам (даёт ускорение в ~2 раза)
///
void MatrixCOOtoCSR( const std::vector<double> &coo_val, const std::vector<int> &coo_row,
    const std::vector<int> &coo_col, std::vector<double> &csr_val, std::vector<int> &csr_first,
    std::vector<int> &csr_kk, bool sorted = false );
//void MatrixCOOtoCSR( const std::vector<double> &coo_val, const std::vector<unsigned> &coo_row,
//    const std::vector<unsigned> &coo_col, std::vector<double> &csr_val, std::vector<unsigned> &csr_first,
//    std::vector<unsigned> &csr_kk, bool sorted = false );

///
/// \brief Преобразование матрицы в COO формате (Coordinate list) в CSR формат (Compressed Sparse Row Yale format)
/// \param[in]  COO - матрица в COO формате (Coordinate list)
/// \param[out] CSR - матрица в CSR формате (Compressed Sparse Row Yale format)
/// \param[in]  sorted - признак отсортированности COO входа по строкам (даёт ускорение в ~2 раза)
///
template<typename T>
void MatrixCOOtoCSR( const CMatrixCOO<T> &COO, CMatrixCSR<T> &CSR, bool sorted = false )
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    MatrixCOOtoCSR( COO.coo_val, COO.coo_row, COO.coo_col, CSR.csr_val, CSR.csr_first, CSR.csr_kk, sorted );
}

///
/// \brief Преобразование матрицы в COO формате (Coordinate list) в CSC формат (Compressed Sparse Column Yale format)
/// \param[in]  coo_val   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  coo_row   - индексы строк ненулевых элементов, размер nnz
/// \param[in]  coo_col   - индексы столбцов ненулевых элементов, размер nnz
/// \param[in]  csc_val   - вектор ненулевых элементов матрицы A, размер равен количеству ненулевых элементов nnz
/// \param[in]  csc_first - вектор индексов строк ненулевых элементов, размер равен количеству ненулевых элементов nnz
/// \param[in]  csc_kk    - вектор начальных смещений в векторе CSR, размер m+1
/// \param[in]  sorted - признак отсортированности COO входа по столбцам (даёт ускорение в ~2 раза)
///
void MatrixCOOtoCSC( const std::vector<double> &coo_val, const std::vector<int> &coo_row,
    const std::vector<int> &coo_col, std::vector<double> &csc_val, std::vector<int> &csc_first,
    std::vector<int> &csc_kk, bool sorted = false );
//void MatrixCOOtoCSC( const std::vector<double> &coo_val, const std::vector<unsigned> &coo_row,
//    const std::vector<unsigned> &coo_col, std::vector<double> &csc_val, std::vector<unsigned> &csc_first,
//    std::vector<unsigned> &csc_kk, bool sorted = false );

///
/// \brief Преобразование матрицы в COO формате (Coordinate list) в CSC формат (Compressed Sparse Column Yale format)
/// \param[in]  COO - матрица в COO формате (Coordinate list)
/// \param[out] CSC - матрица в CSR формате (Compressed Sparse Column Yale format)
/// \param[in]  sorted - признак отсортированности COO входа по столбцам (даёт ускорение в ~2 раза)
///
template<typename T>
void MatrixCOOtoCSC( const CMatrixCOO<T> &COO, CMatrixCSC<T> &CSC, bool sorted = false )
{
    static_assert(
        std::is_same<T, int>::value ||
        std::is_same<T, unsigned>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value,
        "Wrong template class!" );
    MatrixCOOtoCSC( COO.coo_val, COO.coo_row, COO.coo_col, CSC.csc_val, CSC.csc_first, CSC.csc_kk, sorted );
}

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
//    CKeyCOO( unsigned i, unsigned j ) : i_( i ), j_( j ){}

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
//    unsigned i_; ///< Индекс строки
//    unsigned j_; ///< Индекс столбца
};

////----------------------------------------------------------------------------------------------------------------------
/////
///// \brief The CColumnCOO struct
/////
//struct CColumnCOO
//{
//    unsigned i_;
//    unsigned j_;
//    double val_;

//    CColumnCOO() : i_( 0 ), j_( 0 ), val_( 0 ){}

//    CColumnCOO( unsigned i, unsigned j, double val ) : i_( i ), j_( j ), val_( val ){}
//};

} // end namespace Sparse
} // end namespace SPML
#endif // SPML_SPARSE_H
/// \}
