import secrets
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["rsa_keygen", "rsa_encrypt", "rsa_decrypt"]


def is_prime(n: int, k: int = 5) -> bool:
    """Miller–Rabin probabilistic primality test."""
    if n <= 1:
        return False
    # Small primes check
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        if n % p == 0:
            return n == p

    # Write n - 1 as d * 2**s
    s, d = 0, n - 1
    while d & 1 == 0:
        d >>= 1
        s += 1

    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2  # in [2, n-2]
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_prime(bits: int = 1024) -> int:
    """Generate a prime number of specified bit length."""
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(candidate):
            logger.debug("Generated prime of %d bits", bits)
            return candidate


def ex_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm. Returns (g, x, y) with ax + by = g = gcd(a, b)."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = ex_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def modinv(a: int, m: int) -> int:
    """Modular inverse of a modulo m."""
    g, x, _ = ex_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m}")
    return x % m


def rsa_keygen(bits: int = 2048) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Generate RSA key pair.
    Returns ((n, e), (n, d)).
    """
    p = generate_prime(bits // 2)
    q = generate_prime(bits // 2)
    while q == p:
        q = generate_prime(bits // 2)

    n = p * q
    phi = (p - 1) * (q - 1)

    e = 65537
    if math.gcd(e, phi) != 1:
        raise ValueError("e and phi(n) are not coprime")

    d = modinv(e, phi)
    logger.info("RSA key pair generated with modulus %d bits", bits)
    return (n, e), (n, d)


def rsa_encrypt(m: int, public_key: tuple[int, int]) -> int:
    """Encrypt integer m with public key (n, e)."""
    n, e = public_key
    if m < 0 or m >= n:
        raise ValueError("Message out of range")
    return pow(m, e, n)


def rsa_decrypt(c: int, private_key: tuple[int, int]) -> int:
    """Decrypt integer c with private key (n, d)."""
    n, d = private_key
    if c < 0 or c >= n:
        raise ValueError("Ciphertext out of range")
    return pow(c, d, n)


if __name__ == "__main__":
    pub, priv = rsa_keygen(2048)
    print(f"Public key: {pub}\nPrivate key: {priv}")

    msg = 123456789
    ct = rsa_encrypt(msg, pub)
    pt = rsa_decrypt(ct, priv)
    print(f"Message: {msg}\nCipher: {ct}\nDecrypted: {pt}\nValid: {pt == msg}")
