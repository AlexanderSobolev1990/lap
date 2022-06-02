//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap.h
/// \brief      Решение задачи о назначениях (cтандартная линейная дискретная оптимизационная задача)
/// \details    Скорость работы алгоритмов в порядке возрастания: Hungarian (самый медленный), Mack, JVC (самый быстрый)
/// \date       14.07.20 - создан
/// \author     Соболев А.А.
/// \addtogroup spml
/// \{
///

#ifndef SPML_LAP_H
#define SPML_LAP_H

// System includes:
#include <limits>
#include <cassert>
#include <armadillo>

// SPML includes:
#include <sparse.h>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace LAP /// Решение задачи о назначениях
{

///
/// \brief Критерий поиска - минимум/максимум для задачи о назначениях
///
enum TSearchParam
{
    SP_Min, ///< Поиск минимума
    SP_Max  ///< Поиск максимума
};

///
/// \brief Способ нахождения пути в алгоритма JVC для ленточных матриц
///
enum TFindPath
{
    FP_1,
    FP_2,
    FP_DYNAMIC
};

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
/// плотных матриц
/// \details Источники:
/// 1)  "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear
///     Assignment Problems," Computing 38, 325-340, 1987 by
///     R. Jonker and A. Volgenant, University of Amsterdam.
/// 2)  https://github.com/yongyanghz/LAPJV-algorithm-c
/// 3)  https://www.mathworks.com/matlabcentral/fileexchange/26836-lapjv-jonker-volgenant-algorithm-for-linear-assignment-problem-v3-0
/// \remarks    - Оригинальный код R. Jonker and A. Volgenant [1] для целых чисел адаптирован под вещественные
///             - Метод подразделен на 4 процедуры в соответствии с модификацией Castanon:
///                 - COLUMN REDUCTION
///                 - REDUCTION TRANSFER
///                 - AUGMENTING ROW REDUCTION - аукцион
///                 - AUGMENT SOLUTION FOR EACH FREE ROW на основе алгоритма Dijkstra
///             - Правки из [3] касающиеся точности сравнения вещественных чисел
/// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
/// \param[in]  dim        - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
///
void JVCdense( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution,
    arma::ivec &rowsol, double &lapcost );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
/// разреженных матриц
/// \param[in]  cc         -
/// \param[in]  kk         -
/// \param[in]  ii         -
/// \param[in]  dim        - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
/// \param[in]  fp         - find path mode (FP_DYNAMIC - default)
///
//void JVCsparse( const std::vector<double> &cc, const std::vector<unsigned> &kk, const std::vector<unsigned> &ii,
//    unsigned dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost,
//    TFindPath fp = TFindPath::FP_1 );
void JVCsparse( const std::vector<double> &cc, const std::vector<int> &kk, const std::vector<int> &ii,
    unsigned dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost,
    TFindPath fp = TFindPath::FP_1 );

int JVCsparseNEW( const std::vector<double> &cc, const std::vector<int> &kk, const std::vector<int> &first,
    TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost );

///
/// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
/// разреженных матриц
/// \param[in]  csr        - матрица в CSR формате (Compressed Sparse Row Yale format)
/// \param[in]  dim        - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
/// \param[in]  fp         - find path mode (FP_DYNAMIC - default)
///
void JVCsparse( const SPML::Sparse::CMatrixCSR<double> &csr, unsigned dim, TSearchParam sp, double maxcost,
    double resolution, arma::ivec &rowsol, double &lapcost, TFindPath fp = TFindPath::FP_DYNAMIC );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Метод Мака решения задачи о назначениях
/// \details Источник: Банди Б. Основы линейного программирования: Пер. с англ. - М.:Радио м связь, 1989, стр 113-123
/// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
/// \param[in]  dim - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in]  sp - критерий поиска (минимум/максимум)
/// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
///
void Mack( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol,
    double &lapcost );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Венгерский метод решения задачи о назначениях (Метод Мункреса)
/// \details Источник: https://github.com/RcppCore/rcpp-gallery/blob/gh-pages/src/2013-09-24-minimal-assignment.cpp
/// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
/// \param[in]  dim - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in]  sp - критерий поиска (минимум/максимум)
/// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
///
void Hungarian( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution,
    arma::ivec &rowsol, double &lapcost );

/*
///
/// \brief Класс решения задачи о назначениях
///
class CAssignmentProblemSolver
{
public:
    //------------------------------------------------------------------------------------------------------------------
    ///
    /// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
    /// плотных матриц
    /// \details Источники:
    /// 1)  "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear
    ///     Assignment Problems," Computing 38, 325-340, 1987 by
    ///     R. Jonker and A. Volgenant, University of Amsterdam.
    /// 2)  https://github.com/yongyanghz/LAPJV-algorithm-c
    /// 3)  https://www.mathworks.com/matlabcentral/fileexchange/26836-lapjv-jonker-volgenant-algorithm-for-linear-assignment-problem-v3-0
    /// \remarks    - Оригинальный код R. Jonker and A. Volgenant [1] для целых чисел адаптирован под вещественные
    ///             - Метод подразделен на 4 процедуры в соответствии с модификацией Castanon:
    ///                 - COLUMN REDUCTION
    ///                 - REDUCTION TRANSFER
    ///                 - AUGMENTING ROW REDUCTION - аукцион
    ///                 - AUGMENT SOLUTION FOR EACH FREE ROW на основе алгоритма Dijkstra
    ///             - Правки из [3] касающиеся точности сравнения вещественных чисел
    /// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
    /// \param[in]  dim        - порядок квадратной матрицы ценности и размерность результата res соответственно
    /// \param[in]  sp         - критерий поиска (минимум/максимум)
    /// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
    /// \param[in]  resolution - точность для сравнения двух вещественных чисел
    /// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
    /// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
    /// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
    ///
    static void JVCdense( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution,
        arma::ivec &rowsol, double &lapcost );

    //------------------------------------------------------------------------------------------------------------------
    ///
    /// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
    /// разреженных матриц
    /// \param[in]  cc         -
    /// \param[in]  ii         -
    /// \param[in]  kk         -
    /// \param[in]  dim        - порядок квадратной матрицы ценности и размерность результата res соответственно
    /// \param[in]  sp         - критерий поиска (минимум/максимум)
    /// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
    /// \param[in]  resolution - точность для сравнения двух вещественных чисел
    /// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
    /// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
    /// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
    ///
    static void JVCsparse( const std::vector<double> &cc, const std::vector<unsigned> &ii, const std::vector<unsigned> &kk,
        unsigned dim, TSearchParam sp, TFindPath fp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost );

    //------------------------------------------------------------------------------------------------------------------
    ///
    /// \brief Метод Мака решения задачи о назначениях
    /// \details Источник: Банди Б. Основы линейного программирования: Пер. с англ. - М.:Радио м связь, 1989, стр 113-123
    /// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
    /// \param[in]  dim - порядок квадратной матрицы ценности и размерность результата res соответственно
    /// \param[in]  sp - критерий поиска (минимум/максимум)
    /// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
    /// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
    ///
    static void Mack( const arma::mat &assigncost, int dim, TSearchParam sp, arma::ivec &rowsol, double &lapcost );

    //------------------------------------------------------------------------------------------------------------------
    ///
    /// \brief Венгерский метод решения задачи о назначениях (Метод Мункреса)
    /// \details Источник: https://github.com/RcppCore/rcpp-gallery/blob/gh-pages/src/2013-09-24-minimal-assignment.cpp
    /// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
    /// \param[in]  dim - порядок квадратной матрицы ценности и размерность результата res соответственно
    /// \param[in]  sp - критерий поиска (минимум/максимум)
    /// \param[out] rowsol - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
    /// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
    ///
    static void Hungarian( const arma::mat &assigncost, int dim, TSearchParam sp, arma::ivec &rowsol, double &lapcost );

private:
    //------------------------------------------------------------------------------------------------------------------
    // Методы для JVCsparse

    ///
    /// \brief Column-reduction and reduction transfer for a sparse cost matrix
    ///
    static int jvc_sparse_ccrrt_sparse_( const unsigned n, const std::vector<double> &cc, const std::vector<unsigned> &ii,
        const std::vector<unsigned> &kk, int *free_rows, int *x, int *y, double *v );
    ///
    /// \brief Augmenting row reduction for a sparse cost matrix
    ///
    static int jvc_sparse_carr_sparse( const unsigned n, const std::vector<double> &cc, const std::vector<unsigned> &ii,
        const std::vector<unsigned> &kk, const unsigned n_free_rows, int *free_rows, int *x, int *y, double *v );

    ///
    /// \brief Find columns with minimum d[j] and put them on the SCAN list.
    ///
    static unsigned jvc_sparse_find_sparse_1( const unsigned n, unsigned lo, double *d, int *cols, int *y );

    ///
    /// \brief jvc_sparse_find_sparse_2
    ///
    static int jvc_sparse_find_sparse_2( double *d, int *scan, const unsigned n_todo, int *todo, bool *done );

    ///
    /// \brief Scan all columns in TODO starting from arbitrary column in SCAN and try to decrease d of the TODO
    /// columns using the SCAN column.
    ///
    static int jvc_sparse_scan_sparse_1( const unsigned n, const std::vector<double> &cc, const std::vector<unsigned> &ii,
        const std::vector<unsigned> &kk, unsigned *plo, unsigned *phi, double *d, int *cols, int *pred,
        int *y, double *v );

    ///
    /// \brief Scan all columns in TODO starting from arbitrary column in SCAN and try to decrease d of the TODO
    /// columns using the SCAN column.
    ///
    static int jvc_sparse_scan_sparse_2( const unsigned n, const std::vector<double> &cc, const std::vector<unsigned> &ii,
        const std::vector<unsigned> &kk, unsigned *plo, unsigned *phi, double *d, int *pred,
        bool *done, unsigned *pn_ready, int *ready, int *scan,
        unsigned *pn_todo, int *todo, bool *added, int *y, double *v );

    ///
    /// \brief Single iteration of modified Dijkstra shortest path algorithm as explained in the JV paper.
    /// This version loops over all column indices (some of which might be inf).
    /// \return The closest free column index.
    ///
    static int jvc_sparse_find_path_sparse_1( const unsigned n, const std::vector<double> &cc, const std::vector<unsigned> &ii,
        const std::vector<unsigned> &kk, const int start_i, int *y, double *v, int *pred );

    ///
    /// \brief Single iteration of modified Dijkstra shortest path algorithm as explained in the JV paper.
    /// This version loops over non-inf column indices (which requires some additional bookkeeping).
    /// \return The closest free column index.
    ///
    static int jvc_sparse_find_path_sparse_2( const unsigned n, const std::vector<double> &cc, const std::vector<unsigned> &ii,
        const std::vector<unsigned> &kk, const int start_i, int *y, double *v, int *pred );

    ///
    /// \brief Find path using one of the two find_path variants selected based on sparsity.
    ///
    static int jvc_sparse_find_path_sparse_dynamic( const unsigned n, const std::vector<double> &cc, const std::vector<unsigned> &ii,
        const std::vector<unsigned> &kk, const int start_i, int *y, double *v, int *pred );

    typedef int (*fp_function_t)( const unsigned, std::vector<double> &, std::vector<unsigned> &, std::vector<unsigned> &,
        const int, int *, double *, int *);

    static fp_function_t jvc_sparse_get_better_find_path( const unsigned n, const std::vector<unsigned> &ii );

    ///
    /// \brief Augment for a sparse cost matrix.
    ///
    static int jvc_sparse_ca_sparse( const unsigned n, const std::vector<double> &cc, const std::vector<unsigned> &ii,
        const std::vector<unsigned> &kk, const unsigned n_free_rows, int *free_rows, int *x, int *y,
        double *v, int fp_version );

    //------------------------------------------------------------------------------------------------------------------
    // Методы для Hungarian
    static void hungarian_step_1( unsigned int &step, arma::mat &cost,const unsigned int &N );
    static void hungarian_step_2( unsigned int &step, const arma::mat &cost, arma::umat &indM, arma::ivec &rcov,
        arma::ivec &ccov, const unsigned int &N);
    static void hungarian_step_3( unsigned int &step, const arma::umat &indM, arma::ivec &ccov, const unsigned int &N );
    static void hungarian_find_noncovered_zero( int &row, int &col, const arma::mat &cost, const arma::ivec &rcov,
        const arma::ivec &ccov, const unsigned int &N );
    static bool hungarian_star_in_row( int &row, const arma::umat &indM, const unsigned int &N );
    static void hungarian_find_star_in_row( const int &row, int &col, const arma::umat &indM, const unsigned int &N );
    static void hungarian_step_4( unsigned int &step, const arma::mat &cost, arma::umat &indM, arma::ivec &rcov,
        arma::ivec &ccov, int &rpath_0, int &cpath_0, const unsigned int &N );
    static void hungarian_find_star_in_col( const int &col, int &row, const arma::umat &indM, const unsigned int &N );
    static void hungarian_find_prime_in_row( const int &row, int &col, const arma::umat &indM, const unsigned int &N );
    static void hungarian_augment_path( const int &path_count, arma::umat &indM, const arma::imat &path );
    static void hungarian_clear_covers( arma::ivec &rcov, arma::ivec &ccov );
    static void hungarian_erase_primes( arma::umat &indM, const unsigned int &N );
    static void hungarian_step_5( unsigned int &step, arma::umat &indM, arma::ivec &rcov, arma::ivec &ccov,
        arma::imat &path, int &rpath_0, int &cpath_0, const unsigned int &N );
    static void hungarian_find_smallest( double &minval, const arma::mat &cost, const arma::ivec &rcov,
        const arma::ivec &ccov, const unsigned int &N );
    static void hungarian_step_6( unsigned int &step, arma::mat &cost, const arma::ivec &rcov, const arma::ivec &ccov,
        const unsigned int &N );
};
*/



} // end namespace LAP
} // end namespace SPML
#endif // SPML_LAP_H
/// \}
