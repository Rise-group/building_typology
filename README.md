<img src="figs/PEAKurban.png" alt="PEAK Urban logo" align="right" width ="200" height="133">

<img src="figs/logo_rise_eafit.png" alt="RiSE-group logo" align="middle" width ="380" height="100">


Automatic detection of building typology using deep learning methods on street level images
===========================================================================================


## Description

This repository contains all supplementary data that were used in the paper:

#### "Automatic detection of building typology using deep learning methods on street level images"

Daniela Gonzalez<sup>1</sup>, Diego Rueda-Plata<sup>2</sup>, Ana B. Acevedo<sup>1</sup>, Juan C.  Duque<sup>3</sup>, Raúl Ramos-Pollán<sup>4</sup>, Alejandro Betancourt<sup>3</sup> and Sebastian García<sup>3</sup>

<sup>1</sup> Department of Civil Engineering. Universidad EAFIT. Medellín, Colombia.

<sup>2</sup> Universidad Industrial de Santander. Bucaramanga, Colombia.

<sup>3</sup> Research in Spatial Economics (RiSE) Group. Universidad EAFIT. Medellín, Colombia.

<sup>4</sup> Universidad de Antioquia. Medellín, Colombia.


__maintainer__ = "RiSE Group"  (http://www.rise-group.org/). Universidad EAFIT

__Corresponding author__ = jduquec1@eafit.edu.co (JCD)

### Abstract 

An exposure model is a key component for assessing potential human and economic losses from natural disasters. An exposure model consists of a spatially disaggregated description of the infrastructure and population of a region under study. Depending on the size of the settlement area, developing such models can be a costly and time-consuming task. In this paper we use a manually annotated dataset consisting of approximately 10,000 photos acquired at street level in the urban area of Medellín to explore the potential for using a convolutional neural network (CNN) to automatically detect building materials and types of lateral-load resisting systems, which are attributes that define a building’s structural typology (which is a key issue in exposure models for seismic risk assessment). The results of the developed model achieved a precision of 93% and a recall of 95% when identifying nonductile buildings, which are the buildings most likely to be damaged in an earthquake.  Identifying fine-grained material typology is more difficult because many visual clues are physically hidden, but our model matches expert level performances, achieving a recall of 85% and accuracy scores ranging from 60% to 82% on the three most common building typologies, which account for 91% of the total building population in Medellín. Overall, this study shows that a CNN can make a substantial contribution to developing cost-effective exposure models.


## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

## Bibtext entry


```tex
@article{XX,
    author = {Gonzales, D. AND Rueda-Plata, D. AND Acevedo, A. B. AND Duque, J. C. AND Ramos-Pollán, R. AND Betancur, A. AND García, S.},
    journal = {Building and Environment},
    publisher = {Elsevier},
    title = {Automatic detection of building typology using deep learning methods on street level images},
    year = {2020},
    month = {mm},
    volume = {vv},
    url = {xx},
    pages = {},
    abstract = {An exposure model is a key component for assessing potential human and economic losses from natural disasters. An exposure model consists of a spatially disaggregated description of the infrastructure and population of a region under study. Depending on the size of the settlement area, developing such models can be a costly and time-consuming task. In this paper we use a manually annotated dataset consisting of approximately 10,000 photos acquired at street level in the urban area of Medellín to explore the potential for using a convolutional neural network (CNN) to automatically detect building materials and types of lateral-load resisting systems, which are attributes that define a building’s structural typology (which is a key issue in exposure models for seismic risk assessment). The results of the developed model achieved a precision of 93% and a recall of 95% when identifying nonductile buildings, which are the buildings most likely to be damaged in an earthquake.  Identifying fine-grained material typology is more difficult, because many visual clues are physically hidden, but our model matches expert level performances, achieving a recall of 85% and accuracy scores ranging from 60% to 82% on the three most common building typologies, which account for 91% of the total building population in Medellín. Overall, this study shows that a CNN can make a substantial contribution to developing cost-effective exposure models.},
    number = {nn},
    doi = {xx}
}
```
