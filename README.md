# DocketChalllenge [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-388/)


## Proposta
Extrair informações do documento CNH(Carteira Nacional de Habilitação) utilizando processamento de imagem e/ou processamento de texto.

[Notebook Explicativo](https://github.com/AkiraG/DocketChalllenge/blob/main/jupyter/Docket%20CNH.ipynb)

Caso veja necessário rodar novamente o notebook explicativo, deverá instalar a [Object Detection Tensorflow API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)


## Script
Eu encapsulei a aplicação em um script denominado find_cpf.py que recebe um argumento --imageDir, esse busca todas as imagens no diretório informado e retorna o nome da imagem com o cpf

Exemplo:
```
python find_cpf.py --imageDir path/to/images
```
Retorno no prompt

```
Image Name: xx.jpg - CPF : xxx.xxx.xxx-xx
```
Caso a aplicação não consiga encontrar o CPF, retornará o valor None.


### Dependências

Além de instalar as depêndencias contidas no arquivo ```requirements.txt```, é necessario instalar a engine de OCR Tesseract. Por isso é recomendado que seja utilizado o OS Linux (Ubuntu/Debian)
```
apt-get install tesseract-ocr-por
```

Porém existe uma versão para Windows10 do tesseract neste [tutorial](https://medium.com/quantrium-tech/installing-and-using-tesseract-4-on-windows-10-4f7930313f82)


