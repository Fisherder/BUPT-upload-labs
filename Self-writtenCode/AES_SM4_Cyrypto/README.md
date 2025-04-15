# AES and SM4 Crypto

代码支持了 AES 和 SM4 两种加密算法。并且在项目中提供了简单易用的 Java API，方便开发者在应用程序中集成加密功能。

### 功能模块

#### 1. AES 加密模块（`AesCryptoUtils.java`）
AES（高级加密标准）是一种对称加密算法，广泛应用于数据加密。本模块提供了基于 AES 的加密和解密功能，支持 CBC 模式和 PKCS5Padding 填充方式。

- **加密方法**
  
  ```java
  public static String encrypt(String plainText, String key, String iv) throws Exception
  ```
  - **参数**
    - `plainText`：明文字符串，需要被加密的数据。
    - `key`：加密密钥，长度必须为 16 字节（128 位）。
    - `iv`：初始化向量（可选），长度必须为 16 字节。如果未提供，则使用默认的全零向量。
  - **返回值**
    - 加密后的密文，以 Base64 编码的字符串形式返回。
  
- **解密方法**
  ```java
  public static String decrypt(String encryptedText, String key, String iv) throws Exception
  ```
  - **参数**
    - `encryptedText`：密文字符串，需要被解密的数据。
    - `key`：解密密钥，长度必须为 16 字节（128 位）。
    - `iv`：初始化向量（可选），长度必须为 16 字节。如果未提供，则使用默认的全零向量。
  - **返回值**
    - 解密后的明文字符串。

- **密钥和初始化向量校验**
  ```java
  private static void validateKeyIV(byte[] key, byte[] iv)
  ```
  - **功能**
    - 校验密钥和初始化向量的长度是否符合要求。AES 密钥长度必须为 16 字节，初始化向量长度也必须为 16 字节。

#### 2. SM4 加密模块（`Sm4CryptoUtils.java`）
SM4 是我国自主设计的分组密码算法，具有高效性和安全性。本模块提供了基于 SM4 的加密和解密功能，支持 CBC 模式和 PKCS5Padding 填充方式。

- **加密方法**
  ```java
  public static String encrypt(String plainText, String key, String iv) throws Exception
  ```
  - **参数**
    - `plainText`：明文字符串，需要被加密的数据。
    - `key`：加密密钥，长度必须为 16 字节（128 位）。
    - `iv`：初始化向量（可选），长度必须为 16 字节。如果未提供，则使用默认的全零向量。
  - **返回值**
    - 加密后的密文，以 Base64 编码的字符串形式返回。

- **解密方法**
  ```java
  public static String decrypt(String encryptedText, String key, String iv) throws Exception
  ```
  - **参数**
    - `encryptedText`：密文字符串，需要被解密的数据。
    - `key`：解密密钥，长度必须为 16 字节（128 位）。
    - `iv`：初始化向量（可选），长度必须为 16 字节。如果未提供，则使用默认的全零向量。
  - **返回值**
    - 解密后的明文字符串。

- **密钥和初始化向量校验**
  ```java
  private static void validateKeyIV(byte[] key, byte[] iv)
  ```
  - **功能**
    - 校验密钥和初始化向量的长度是否符合要求。SM4 密钥长度必须为 16 字节，初始化向量长度也必须为 16 字节。

### 使用示例

#### AES 加密解密示例
```java
import com.example.cryptotool.utils.AesCryptoUtils;

public class AesExample {
    public static void main(String[] args) {
        try {
            String plainText = "Hello, AES!";
            String key = "0123456789abcdef"; // 16 字节密钥
            String iv = "fedcba9876543210"; // 16 字节初始化向量

            // 加密
            String encryptedText = AesCryptoUtils.encrypt(plainText, key, iv);
            System.out.println("Encrypted Text: " + encryptedText);

            // 解密
            String decryptedText = AesCryptoUtils.decrypt(encryptedText, key, iv);
            System.out.println("Decrypted Text: " + decryptedText);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

#### SM4 加密解密示例
```java
import com.example.cryptotool.utils.Sm4CryptoUtils;

public class Sm4Example {
    public static void main(String[] args) {
        try {
            String plainText = "Hello, SM4!";
            String key = "0123456789abcdef"; // 16 字节密钥
            String iv = "fedcba9876543210"; // 16 字节初始化向量

            // 加密
            String encryptedText = Sm4CryptoUtils.encrypt(plainText, key, iv);
            System.out.println("Encrypted Text: " + encryptedText);

            // 解密
            String decryptedText = Sm4CryptoUtils.decrypt(encryptedText, key, iv);
            System.out.println("Decrypted Text: " + decryptedText);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### 依赖说明
代码依赖于 Bouncy Castle 提供的加密库，用于支持 SM4 算法。在使用 SM4 功能之前，请确保已添加 Bouncy Castle 提供者。

```java
Security.addProvider(new BouncyCastleProvider());
```

### 注意事项
1. **密钥和初始化向量长度**：AES 和 SM4 的密钥长度必须为 16 字节（128 位），初始化向量长度也必须为 16 字节。
2. **安全性**：请妥善保管密钥和初始化向量，避免泄露。
3. **异常处理**：在实际使用中，请对加密和解密操作进行异常处理，确保程序的健壮性。

