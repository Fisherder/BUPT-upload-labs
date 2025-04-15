# RSA加密算法

## 项目简介

代码使用Python，实现了RSA算法，并提供了些许测试样例。

## 功能介绍

**生成RSA秘钥**

```python
def rsa_keygen(bits: int = 2048) -> tuple:
```
函数接受一个参数``bits``决定生成密钥的位数（默认为2048位），并返回一个包含公钥公钥``(n, e)``和私钥``(n, d)``的元组。

**加密**

```python
def rsa_encrypt(m: int, public_key: tuple) -> int:
```
函数接收整数形式的原文``m``以及公钥元组``public_key``，返回整数形式的密文。

**解密**

```
def rsa_decrypt(c: int, private_key: tuple) -> int:
```

函数接受整数形式的密文``c``以及私钥元组``private_key``,返回整数形式的原文。
