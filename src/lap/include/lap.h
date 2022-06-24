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
#include <set>

// SPML includes:
#include <sparse.h>

namespace SPML /// Специальная библиотека программных модулей (СБ ПМ)
{
namespace LAP /// Решение задачи о назначениях
{
//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Критерий поиска - минимум/максимум для задачи о назначениях
///
enum TSearchParam
{
    SP_Min, ///< Поиск минимума
    SP_Max  ///< Поиск максимума
};

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Процедура последовательного поиска экстремума по строкам/столбцам матрицы ценностей
/// \details Неоптимальная процедура, сумма назначений будет неоптимальна, поскольку строки/столбцы с найденным
/// экстремумом исключаются из анализа.
/// \param[in]  assigncost - матрица ценности
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
///
void SequentalExtremum( const arma::mat &assigncost, TSearchParam sp, double maxcost, double resolution,
    arma::ivec &rowsol, double &lapcost );

void SequentalExtremum( const Sparse::CMatrixCOO assigncost, TSearchParam sp, double maxcost, double resolution,
    arma::ivec &rowsol, double &lapcost );

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
/// \param[in]  cc         - вектор ненулевых элементов матрицы
/// \param[in]  kk         - вектор индексов колонок ненулевых элементов, размер равен количеству ненулевых элементов
/// \param[in]  first      - вектор начальных смещений в векторе сс
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
/// \return 0 - OK, 1 - fail
///
int JVCsparse( const std::vector<double> &cc, const std::vector<int> &kk, const std::vector<int> &first,
    TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol, double &lapcost );

///
/// \brief Метод Джонкера-Волгенанта-Кастаньона (Jonker-Volgenant-Castanon) решения задачи о назначениях для
/// разреженных матриц
/// \param[in]  csr        - матрица в CSR формате (Compressed Sparse Row Yale format)
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
/// \return 0 - OK, 1 - fail
///
int JVCsparse( const Sparse::CMatrixCSR &csr, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol,
    double &lapcost );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Метод Мака решения задачи о назначениях
/// \details Источник: Банди Б. Основы линейного программирования: Пер. с англ. - М.:Радио м связь, 1989, стр 113-123
/// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
/// \param[in]  dim        - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
///
void Mack( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution, arma::ivec &rowsol,
    double &lapcost );

//----------------------------------------------------------------------------------------------------------------------
///
/// \brief Венгерский метод решения задачи о назначениях (Метод Мункреса)
/// \details Источник: https://github.com/RcppCore/rcpp-gallery/blob/gh-pages/src/2013-09-24-minimal-assignment.cpp
/// \param[in]  assigncost - квадратная матрица ценности, размер [dim,dim]
/// \param[in]  dim        - порядок квадратной матрицы ценности и размерность результата res соответственно
/// \param[in]  sp         - критерий поиска (минимум/максимум)
/// \param[in]  maxcost    - модуль максимального элемента матрицы assigncost
/// \param[in]  resolution - точность для сравнения двух вещественных чисел
/// \param[out] rowsol     - результат задачи о назначениях, размерность [dim] (индекс макс/мин элемента в i-ой строке)
/// rowsol[i] = j --> в i-ой строке выбран j-ый элемент
/// \param[out] lapcost    - сумма назначенных элементов матрицы ценности assigncost
///
void Hungarian( const arma::mat &assigncost, int dim, TSearchParam sp, double maxcost, double resolution,
    arma::ivec &rowsol, double &lapcost );

} // end namespace LAP
} // end namespace SPML
#endif // SPML_LAP_H
/// \}
