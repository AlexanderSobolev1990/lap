//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap.h
/// \brief      Решение задачи о назначениях (cтандартная линейная дискретная оптимизационная задача)
/// \details    Скорость работы алгоритмов в порядке возрастания: Hungarian (самый медленный), Mack, JVC (самый быстрый)
/// \date       14.07.20 - создан
/// \author     Соболев А.А.
///

#ifndef LAP_H
#define LAP_H

#include <limits>
#include <armadillo>

namespace LAP /// Решение задачи о назначениях
{
//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Критерий поиска - минимум/максимум для задачи о назначениях
///
enum TSearchParam
{
    Min, ///< Поиск минимума
    Max  ///< Поиск максимума
};

///
/// \brief Класс решения задачи о назначениях
///
class CAssignmentProblemSolver
{
public:
    //------------------------------------------------------------------------------------------------------------------
    ///
    /// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях
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
    /// \param[out] rowsol - результат задачи о назначениях, размерность [dim]
    ///
    static void JVC( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution,
        arma::ivec &rowsol );

    //------------------------------------------------------------------------------------------------------------------
    ///
    /// \brief Метод Мака решения задачи о назначениях
    /// \details Взят из: Банди Б. Основы линейного программирования: Пер. с англ. - М.:Радио м связь, 1989, стр 113-123
    /// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
    /// \param[in]  dim - порядок квадратной матрицы ценности и размерность результата res соответственно
    /// \param[in]  sp - критерий поиска (минимум/максимум)
    /// \param[out] rowsol - результат задачи о назначениях, размерность [dim]
    ///
    static void Mack( const arma::mat &assigncost, int dim, TSearchParam sp, arma::ivec &rowsol );

    //------------------------------------------------------------------------------------------------------------------
    ///
    /// \brief Венгерский метод решения задачи о назначениях (Метод Мункреса)
    /// \details Взят из: https://github.com/RcppCore/rcpp-gallery/blob/gh-pages/src/2013-09-24-minimal-assignment.cpp
    /// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
    /// \param[in]  dim - порядок квадратной матрицы ценности и размерность результата res соответственно
    /// \param[in]  sp - критерий поиска (минимум/максимум)
    /// \param[out] rowsol - результат задачи о назначениях, размерность [dim]
    ///
    static void Hungarian( const arma::mat &assigncost, int dim, TSearchParam sp, arma::ivec &rowsol );

private:
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

}
#endif // LAP_H
