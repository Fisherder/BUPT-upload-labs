import { sm2 } from 'sm-crypto-v2';

// SM2 加密解密测试
describe('SM2 Encryption and Decryption', () => {
    const keypair = sm2.generateKeyPairHex();
    const publicKey = keypair.publicKey;
    const privateKey = keypair.privateKey;

    const msgString = 'Hello, SM2!';
    const msgArray = new TextEncoder().encode(msgString); // 将字符串转换为 Uint8Array

    test('should encrypt and decrypt the message correctly', () => {
        const encrypted = sm2.doEncrypt(msgString, publicKey);
        expect(encrypted).toBeDefined();
        expect(typeof encrypted).toBe('string');

        const decrypted = sm2.doDecrypt(encrypted, privateKey);
        expect(decrypted).toBe(msgString);
    });

    test('should encrypt and decrypt array input correctly', () => {
        const cipherMode = 1; // 1 - C1C3C2
        const msgArray = new TextEncoder().encode('hello world');

        // 生成一个新的密钥对
        const keypair = sm2.generateKeyPairHex();
        const publicKey = keypair.publicKey;  // 获取生成的公钥
        const privateKey = keypair.privateKey;  // 获取生成的私钥

        // 加密数据，使用指定的 cipherMode
        const encrypted = sm2.doEncrypt(msgArray, publicKey, cipherMode);
        expect(encrypted).toBeDefined();
        expect(typeof encrypted).toBe('string');

        // 解密时确保传递 cipherMode 和正确的输出选项
        const decrypted = sm2.doDecrypt(encrypted, privateKey, cipherMode, { output: 'array' });

        // Debug output: check the type and value of decrypted
        console.log('Decrypted output:', decrypted);
        console.log('Type of decrypted:', decrypted instanceof Uint8Array ? 'Uint8Array' : typeof decrypted);

        // 检查解密结果是否为 Uint8Array 实例
        expect(decrypted).toBeInstanceOf(Uint8Array);

        // 使用 TextDecoder 解码
        const decodedString = new TextDecoder().decode(decrypted);
        expect(decodedString).toBe('hello world');
    });

    test('should fail decryption with wrong key', () => {
        const encrypted = sm2.doEncrypt(msgString, publicKey);
        const wrongKeypair = sm2.generateKeyPairHex();
        const decrypted = sm2.doDecrypt(encrypted, wrongKeypair.privateKey);
        expect(decrypted).not.toBe(msgString);
    });

    test('should compress and decompress public key correctly', () => {
        const compressedPublicKey = sm2.compressPublicKeyHex(publicKey);
        expect(compressedPublicKey).toBeDefined();

        const isEquivalent = sm2.comparePublicKeyHex(publicKey, compressedPublicKey);
        expect(isEquivalent).toBe(true);
    });

    test('should verify public key correctly', () => {
        const verifyResult = sm2.verifyPublicKey(publicKey);
        expect(verifyResult).toBe(true);

        const compressedPublicKey = sm2.compressPublicKeyHex(publicKey);
        const verifyResultCompressed = sm2.verifyPublicKey(compressedPublicKey);
        expect(verifyResultCompressed).toBe(true);
    });
});

// SM2 签名验签测试
describe('SM2 Signing and Verification', () => {
    const keypair = sm2.generateKeyPairHex();
    const publicKey = keypair.publicKey;
    const privateKey = keypair.privateKey;
    const msg = 'Sample message for signing';

    test('should sign and verify the message correctly', () => {
        const sigValueHex = sm2.doSignature(msg, privateKey);
        expect(sigValueHex).toBeDefined();

        const verifyResult = sm2.doVerifySignature(msg, sigValueHex, publicKey);
        expect(verifyResult).toBe(true);
    });

    test('should fail verification with altered message', () => {
        const sigValueHex = sm2.doSignature(msg, privateKey);
        const alteredMsg = 'Altered message for signing';

        const verifyResult = sm2.doVerifySignature(alteredMsg, sigValueHex, publicKey);
        expect(verifyResult).toBe(false);
    });

    test('should sign and verify using precomputed public key', () => {
        const precomputedPublicKey = sm2.precomputePublicKey(publicKey);
        const sigValueHex = sm2.doSignature(msg, privateKey);
        expect(sigValueHex).toBeDefined();

        const verifyResult = sm2.doVerifySignature(msg, sigValueHex, precomputedPublicKey);
        expect(verifyResult).toBe(true);
    });

    test('should sign and verify using various options', () => {
        const sigValueHex = sm2.doSignature(msg, privateKey, { hash: true });
        expect(sigValueHex).toBeDefined();

        const verifyResult = sm2.doVerifySignature(msg, sigValueHex, publicKey, { hash: true });
        expect(verifyResult).toBe(true);
    });

    test('should sign with user ID and verify with user ID', () => {
        const userId = 'testUserId';
        const sigValueHex = sm2.doSignature(msg, privateKey, { userId });
        expect(sigValueHex).toBeDefined();

        const verifyResult = sm2.doVerifySignature(msg, sigValueHex, publicKey, { userId });
        expect(verifyResult).toBe(true);
    });
});

// 测试椭圆曲线点获取
describe('SM2 Point Generation', () => {
    test('should get a valid elliptic curve point', () => {
        const point = sm2.getPoint();
        expect(point).toBeDefined();
    });
});
