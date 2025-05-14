import { sm2 } from 'sm-crypto-v2';

// SM2 加密解密测试
describe('SM2 Encryption and Decryption', () => {
    const globalKeyPair = sm2.generateKeyPairHex(); // 生成全局密钥对
    const globalPublicKey = globalKeyPair.publicKey; // 获取公钥
    const globalPrivateKey = globalKeyPair.privateKey; // 获取私钥

    const msgString = 'Hello, SM2!'; // 要加密的消息
    const msgArray = new TextEncoder().encode(msgString); // 将字符串转换为 Uint8Array

    test('should encrypt and decrypt the message correctly', () => {
        // 使用公钥加密消息
        const encrypted = sm2.doEncrypt(msgString, globalPublicKey);
        expect(encrypted).toBeDefined(); // 确保加密结果已定义
        expect(typeof encrypted).toBe('string'); // 加密后的结果应为字符串

        // 使用私钥解密消息
        const decrypted = sm2.doDecrypt(encrypted, globalPrivateKey);
        expect(decrypted).toBe(msgString); // 解密后的结果应与原消息相同
    });

    test('should encrypt and decrypt array input correctly', () => {
        const cipherMode = 1; // 设置加密模式为 C1C3C2
        const localMsgArray = new TextEncoder().encode('hello world'); // 要加密的消息数组

        // 生成一个新的密钥对
        const localKeyPair = sm2.generateKeyPairHex();
        const localPublicKey = localKeyPair.publicKey;
        const localPrivateKey = localKeyPair.privateKey;

        // 使用指定的 cipherMode 加密消息
        const encrypted = sm2.doEncrypt(localMsgArray, localPublicKey, cipherMode);
        expect(encrypted).toBeDefined(); // 确保加密结果已定义
        expect(typeof encrypted).toBe('string'); // 加密结果应为字符串

        // 解密时使用指定的 cipherMode
        const decrypted = sm2.doDecrypt(encrypted, localPrivateKey, cipherMode, { output: 'array' });

        // 解密结果应为 Uint8Array 实例
        expect(decrypted).toBeInstanceOf(Uint8Array);

        // 使用 TextDecoder 解码 Uint8Array 为字符串
        const decodedString = new TextDecoder().decode(decrypted);
        expect(decodedString).toBe('hello world'); // 解码后的字符串应为原始消息
    });

    test('should fail decryption with wrong key', () => {
        // 使用正确的公钥加密消息
        const encrypted = sm2.doEncrypt(msgString, globalPublicKey);
        const wrongKeypair = sm2.generateKeyPairHex(); // 生成一个不同的密钥对
        const decrypted = sm2.doDecrypt(encrypted, wrongKeypair.privateKey); // 使用错误的私钥解密
        expect(decrypted).not.toBe(msgString); // 解密结果应与原消息不同
    });

    test('should compress and decompress public key correctly', () => {
        // 压缩公钥
        const compressedPublicKey = sm2.compressPublicKeyHex(globalPublicKey);
        expect(compressedPublicKey).toBeDefined(); // 压缩后的公钥应已定义

        // 比较压缩后的公钥和原公钥
        const isEquivalent = sm2.comparePublicKeyHex(globalPublicKey, compressedPublicKey);
        expect(isEquivalent).toBe(true); // 比较结果应为真，说明压缩后的公钥与原公钥等效
    });

    test('should verify public key correctly', () => {
        // 验证公钥
        const verifyResult = sm2.verifyPublicKey(globalPublicKey);
        expect(verifyResult).toBe(true); // 公钥验证应成功

        // 压缩公钥并验证
        const compressedPublicKey = sm2.compressPublicKeyHex(globalPublicKey);
        const verifyResultCompressed = sm2.verifyPublicKey(compressedPublicKey);
        expect(verifyResultCompressed).toBe(true); // 压缩后的公钥验证应成功
    });
});

// SM2 签名验签测试
describe('SM2 Signing and Verification', () => {
    const signKeyPair = sm2.generateKeyPairHex(); // 生成签名密钥对
    const signPublicKey = signKeyPair.publicKey; // 获取签名公钥
    const signPrivateKey = signKeyPair.privateKey; // 获取签名私钥
    const signMsg = 'Sample message for signing'; // 要签名的消息

    test('should sign and verify the message correctly', () => {
        // 使用私钥对消息进行签名
        const sigValueHex = sm2.doSignature(signMsg, signPrivateKey);
        expect(sigValueHex).toBeDefined(); // 签名结果应已定义

        // 使用公钥验证签名
        const verifyResult = sm2.doVerifySignature(signMsg, sigValueHex, signPublicKey);
        expect(verifyResult).toBe(true); // 验证签名应成功
    });

    test('should fail verification with altered message', () => {
        // 使用私钥对消息进行签名
        const sigValueHex = sm2.doSignature(signMsg, signPrivateKey);
        const alteredMsg = 'Altered message for signing'; // 修改消息内容

        // 验证签名时使用修改后的消息
        const verifyResult = sm2.doVerifySignature(alteredMsg, sigValueHex, signPublicKey);
        expect(verifyResult).toBe(false); // 验证应失败
    });

    test('should sign and verify using precomputed public key', () => {
        // 预计算公钥（此步骤加速验证）
        const precomputedPublicKey = sm2.precomputePublicKey(signPublicKey);
        const sigValueHex = sm2.doSignature(signMsg, signPrivateKey);
        expect(sigValueHex).toBeDefined(); // 签名应已定义

        // 使用预计算的公钥验证签名
        const verifyResult = sm2.doVerifySignature(signMsg, sigValueHex, precomputedPublicKey);
        expect(verifyResult).toBe(true); // 验证应成功
    });

    test('should sign and verify using various options', () => {
        // 使用哈希选项进行签名
        const sigValueHex = sm2.doSignature(signMsg, signPrivateKey, { hash: true });
        expect(sigValueHex).toBeDefined(); // 签名应已定义

        // 使用哈希选项验证签名
        const verifyResult = sm2.doVerifySignature(signMsg, sigValueHex, signPublicKey, { hash: true });
        expect(verifyResult).toBe(true); // 验证应成功
    });

    test('should sign with user ID and verify with user ID', () => {
        const userId = 'testUserId'; // 设置用户ID
        const sigValueHex = sm2.doSignature(signMsg, signPrivateKey, { userId });
        expect(sigValueHex).toBeDefined(); // 签名应已定义

        // 使用用户ID验证签名
        const verifyResult = sm2.doVerifySignature(signMsg, sigValueHex, signPublicKey, { userId });
        expect(verifyResult).toBe(true); // 验证应成功
    });
});

// 测试椭圆曲线点获取
describe('SM2 Point Generation', () => {
    test('should get a valid elliptic curve point', () => {
        // 获取椭圆曲线上的一个点
        const point = sm2.getPoint();
        expect(point).toBeDefined(); // 曲线点应已定义
    });
});
