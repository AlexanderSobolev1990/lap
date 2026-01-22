cmake_minimum_required(VERSION 3.7)
set(DOXYGEN_TARGET_NAME project_documentation) # Вторая часть имени файла с настройками Doxygen
get_filename_component(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ ABSOLUTE) # Путь к корневой директории решения (solution)

set(DOXYGEN_INPUT_DIRECTORY # Откуда брать исходники для документации
    ${ROOT_DIR}/src/lap
)
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}) # Куда положить сгенерированную документацию

set(DOXYGEN_IMAGE_PATH ${ROOT_DIR}/doc/pictures)
#set(DOXYGEN_IMAGE_PATH pictures)

if(${CMAKE_VERSION} VERSION_GREATER "3.8")
    message(WARNING "Doxygen do not need config file, options are taken from CMakeLists.txt")

    # Укажем ключи Doxygen
    set(DOXYGEN_DOXYFILE_ENCODING UTF-8)
    set(DOXYGEN_OUTPUT_LANGUAGE RUSSIAN)
    set(DOXYGEN_PROJECT_NAME "LAP (Linear Assignment Problem)")
    set(DOXYGEN_PROJECT_BRIEF "Задача о назначениях (назначить K 'работникам' M 'работ' при минимальных суммарных затратах)")
    set(DOXYGEN_EXTRACT_ALL YES) #NO) #
    set(DOXYGEN_EXTRACT_PRIVATE YES) # NO) #
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${ROOT_DIR}/README.md") #
    set(DOXYGEN_GENERATE_HTML YES)    
    set(DOXYGEN_GENERATE_LEGEND YES)
    #set(DOXYGEN_DOT_GRAPH_MAX_NODES 10)
#    set(DOXYGEN_DOT_GRAPH_DEPTH 2)
    set(DOXYGEN_FULL_PATH_NAMES NO)

    set(DOXYGEN_HAVE_DOT NO) # Попытка выключить диаграммы классов

    set(DOXYGEN_SOURCE_BROWSER YES)#NO)#    # Включать исходники! *.cpp
    set(DOXYGEN_SHOW_FILES YES)#NO)#

    set(DOXYGEN_GENERATE_LATEX YES)
    set(DOXYGEN_USE_PDFLATEX YES) #NO) #
    set(DOXYGEN_PDF_HYPERLINKS YES) # NO) #
    set(DOXYGEN_LATEX_SOURCE_CODE YES) # NO) #
    set(DOXYGEN_LATEX_HIDE_INDICES)
#    set(DOXYGEN_COMPACT_LATEX YES)
    # set(DOXYGEN_LATEX_HEADER ${ROOT_DIR}/doc/forlatex/custom_header.tex) # То, что включается перед осн. частью
#    set(DOXYGEN_LATEX_FOOTER ${ROOT_DIR}/doc/forlatex/custom_footer.tex) # То, что включается после осн. части
#    set(DOXYGEN_LATEX_EXTRA_STYLESHEET ${ROOT_DIR}/doc/forlatex/espd.sty) # Кастомные настройки


    set(DOXYGEN_GENERATE_TESTLIST YES)
    set(DOXYGEN_GENERATE_TODOLIST YES)

    set(DOXYGEN_GENERATE_TREEVIEW YES)
#    set(DOXYGEN_SOURCE_BROWSER NO) # Не включать исходники! *.cpp
    set(DOXYGEN_SOURCE_BROWSER YES) # Включать исходники! *.cpp
    set(DOXYGEN_HTML_TIMESTAMP YES)

    # Создадим файл настроек Doxygen
    doxygen_add_docs(
        ${DOXYGEN_TARGET_NAME}
#        ${ROOT_DIR}/README.md
        ${ROOT_DIR}/doc/forlatex/custom_header.tex
        ${DOXYGEN_INPUT_DIRECTORY}
        ALL
        WORKING_DIRECTORY ${DOXYGEN_OUTPUT_DIRECTORY}
        COMMENT "Generating doxygen project documentation"
    )
else()
    # Версия CMake <= 3.9 НЕ умеет сама создавать файл настроек doxygen, поэтому файл должен быть заранее подготовлен!
    set(DOXYGEN_SETTINGS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.${DOXYGEN_TARGET_NAME}) # CMake дает имя файлу настроек Doxyfile.{user_defined}

    if(EXISTS ${DOXYGEN_SETTINGS_FILE}) # Если файл настроек есть
        add_custom_target(
            ${DOXYGEN_TARGET_NAME}
            ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_SETTINGS_FILE} # doxygen-у надо передать имя файла с настройками
            WORKING_DIRECTORY ${DOXYGEN_OUTPUT_DIRECTORY}
            COMMENT "Generating doxygen project documentation"
        )
    else()
        message(WARNING "Doxygen config file was not found, project documenation creation skipped!")
    endif()
endif()


