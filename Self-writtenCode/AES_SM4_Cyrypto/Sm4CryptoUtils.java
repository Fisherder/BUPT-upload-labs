package com.example.cryptotool.utils;

import org.bouncycastle.jce.provider.BouncyCastleProvider;
import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.Security;
import java.util.Base64;

public class Sm4CryptoUtils {
    private static final String ALGORITHM = "SM4";
    private static final String TRANSFORMATION = "SM4/CBC/PKCS5Padding";
    private static final byte[] DEFAULT_IV = new byte[16]; // 全零向量

    static {
        // 添加BouncyCastle提供者
        Security.addProvider(new BouncyCastleProvider());
    }

    public static String encrypt(String plainText, String key, String iv) throws Exception {
        // 参数校验
        if (plainText == null || key == null) {
            throw new IllegalArgumentException("Plain text and key cannot be null");
        }

        byte[] keyBytes = key.getBytes(StandardCharsets.UTF_8);
        byte[] ivBytes = (iv == null || iv.isEmpty()) ? DEFAULT_IV : iv.getBytes(StandardCharsets.UTF_8);

        validateKeyIV(keyBytes, ivBytes);

        SecretKeySpec secretKey = new SecretKeySpec(keyBytes, ALGORITHM);
        IvParameterSpec ivSpec = new IvParameterSpec(ivBytes);

        Cipher cipher = Cipher.getInstance(TRANSFORMATION, "BC");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey, ivSpec);

        byte[] encryptedBytes = cipher.doFinal(plainText.getBytes(StandardCharsets.UTF_8));
        return Base64.getEncoder().encodeToString(encryptedBytes);
    }

    public static String decrypt(String encryptedText, String key, String iv) throws Exception {
        // 参数校验
        if (encryptedText == null || key == null) {
            throw new IllegalArgumentException("Encrypted text and key cannot be null");
        }

        byte[] keyBytes = key.getBytes(StandardCharsets.UTF_8);
        byte[] ivBytes = (iv == null || iv.isEmpty()) ? DEFAULT_IV : iv.getBytes(StandardCharsets.UTF_8);

        validateKeyIV(keyBytes, ivBytes);

        SecretKeySpec secretKey = new SecretKeySpec(keyBytes, ALGORITHM);
        IvParameterSpec ivSpec = new IvParameterSpec(ivBytes);

        Cipher cipher = Cipher.getInstance(TRANSFORMATION, "BC");
        cipher.init(Cipher.DECRYPT_MODE, secretKey, ivSpec);

        byte[] encryptedBytes = Base64.getDecoder().decode(encryptedText);
        byte[] decryptedBytes = cipher.doFinal(encryptedBytes);

        return new String(decryptedBytes, StandardCharsets.UTF_8);
    }

    private static void validateKeyIV(byte[] key, byte[] iv) {
        if (key.length != 16) {
            throw new IllegalArgumentException("Invalid SM4 key length (must be 16 bytes)");
        }
        if (iv.length != 16) {
            throw new IllegalArgumentException("Invalid IV length (must be 16 bytes)");
        }
    }
} 