//----------------------------------------------------------------------------------------------------------------------
///
/// \file       lap.h
/// \brief      Решение задачи о назначениях (cтандартная линейная дискретная оптимизационная задача)
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
    Min,
    Max
};

//----------------------------------------------------------------------------------------------------------------------
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
void JVC( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Метод Мака решения задачи о назначениях
/// \details Взят из: Банди Б. Основы линейного программирования: Пер. с англ. - М.:Радио м связь, 1989, стр 113-123
/// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
/// \param[in]  dim - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in]  sp - критерий поиска (минимум/максимум)
/// \param[out] rowsol - результат задачи о назначениях, размерность [dim]
/// \warning Значения элементов входной матрицы ценности привязок p не сохраняются!
///
void Mack( const arma::mat &assigncost, int dim, TSearchParam sp, arma::ivec &rowsol );

}
#endif // LAP_H
