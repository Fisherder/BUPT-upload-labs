import random
import math


def _is_prime(n: int, k: int = 5) -> bool:
    """
    Miller-Rabin素数检测算法
    n: 待检测数
    k: 检测轮数
    return: 是否为素数
    """
    if n <= 1:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if n % p == 0:
            return n == p

    # 将n-1分解为d*2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(k):
        a = random.randint(2, min(n - 2, 2**20))
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for __ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int = 1024) -> int:
    """
    生成指定位数的素数
    bits: 素数位数
    return: 生成的素数
    """
    while True:
        p = random.getrandbits(bits)
        p |= (1 << bits - 1) | 1  # 确保最高位为1且为奇数
        if _is_prime(p):
            return p


def _modinv(a: int, m: int) -> int:
    """
    扩展欧几里得算法求模逆元
    return: a⁻¹ mod m
    """
    g, x, y = _ex_gcd(a, m)
    if g != 1:
        raise ValueError("模逆不存在")
    return x % m


def _ex_gcd(a: int, b: int) -> tuple:
    """
    扩展欧几里得算法
    return: (gcd, x, y) 满足 ax + by = gcd(a, b)
    """
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = _ex_gcd(b % a, a)
        return (g, x - (b // a) * y, y)


def rsa_keygen(bits: int = 2048) -> tuple:
    """
    RSA密钥生成函数
    bits: 模数n的位数
    return: (公钥(n, e), 私钥(n, d))
    """
    # 生成两个不同的大素数
    p = _generate_prime(bits // 2)
    q = _generate_prime(bits // 2)
    while p == q:
        q = _generate_prime(bits // 2)

    # 生成n
    n = p * q
    phi = (p - 1) * (q - 1)

    # 选择公钥指数e
    e = 65537  # 常用公钥指数65537
    assert math.gcd(e, phi) == 1, "e与φ(n)不互质，请重新生成密钥"

    # 计算私钥指数d
    d = _modinv(e, phi)

    return ((n, e), (n, d))


def rsa_encrypt(m: int, public_key: tuple) -> int:
    """
    RSA加密
    m: 明文（整数）
    public_key: 公钥(n, e)
    return: 密文c
    """
    n, e = public_key
    if m >= n:
        raise ValueError("原文长度过长")
    return pow(m, e, n)


def rsa_decrypt(c: int, private_key: tuple) -> int:
    """
    RSA解密
    c: 密文
    private_key: 私钥(n, d)
    return: 解密后的明文
    """
    n, d = private_key
    if c >= n:
        raise ValueError("密文无效")
    return pow(c, d, n)


# 测试代码
if __name__ == "__main__":
    # 生成密钥对
    public_key, private_key = rsa_keygen(2048)
    print(f"公钥 (n, e):\n{public_key}")
    print(f"\n私钥 (n, d):\n{private_key}")

    # 测试加密解密
    message = 123456789
    ciphertext = rsa_encrypt(message, public_key)
    decrypted = rsa_decrypt(ciphertext, private_key)

    print(f"\n原始消息: {message}")
    print(f"加密结果: {ciphertext}")
    print(f"解密结果: {decrypted}")
    print(f"加解密验证: {message == decrypted}")
