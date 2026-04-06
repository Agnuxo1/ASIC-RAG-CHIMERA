#!/usr/bin/env python3
"""
EXHAUSTIVE HEADER BYTE-ORDER SEARCH

Uses real captured data from V7 to find the correct header construction.
Tests ALL possible combinations of byte orderings for each field.

Data from: verified_proofs_20251228_121202.json (V7, diff=1)
"""
import hashlib
import struct
import itertools

# =============================================================================
# DATA FROM V7 PROOF (DIFFICULTY=1, nbits=1d00ffff)
# =============================================================================

# Submitted by ASIC
NONCE_HEX = "639c00d8"
NTIME_HEX = "69511058"
VERSION_BITS_HEX = "038f6000"
EXTRANONCE2_HEX = "08000000"

# Job data
MEDICAL_MERKLE_ROOT_HEX = "7aaf846c36b24b1ee7cd807b6a1e6e1761f1187f8dae2a75fc69aa30adec31ed"
COINB1_HEX = "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff1701014348494d4552412f"
COINB2_HEX = "2f4d45442fffffffff0100f2052a01000000126a10415349432d5241472d4845414c54482100000000"
EXTRANONCE1_HEX = "deadbeef"
NBITS_HEX = "1d00ffff"  # Correct for difficulty 1
TARGET_HEX = "00000000ffff0000000000000000000000000000000000000000000000000000"

# What V7 computed (WRONG - hash ratio 1.5B)
V7_HEADER_HEX = "00608f037aaf846c36b24b1ee7cd807b6a1e6e1761f1187f8dae2a75fc69aa30adec31ed7c2d7ff17804e2798433b9f07a3e8a26fda1d602098770ca53436007b4926a1258105169ffff001dd8009c63"
V7_COINBASE_HASH_HEX = "7c2d7ff17804e2798433b9f07a3e8a26fda1d602098770ca53436007b4926a12"

# =============================================================================
# UTILS
# =============================================================================

def hex_to_bytes(h):
    return bytes.fromhex(h)

def bytes_to_hex(b):
    return b.hex()

def reverse_bytes(b):
    return b[::-1]

def sha256d(b):
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()

def swap_words(hex_str):
    """Swap 32-bit words in hex string (Stratum prevhash format)"""
    if len(hex_str) % 8 != 0:
        hex_str = hex_str.zfill((len(hex_str) // 8 + 1) * 8)
    result = ""
    for i in range(0, len(hex_str), 8):
        word = hex_str[i:i+8]
        swapped = "".join(reversed([word[j:j+2] for j in range(0, 8, 2)]))
        result += swapped
    return result

# =============================================================================
# EXHAUSTIVE SEARCH
# =============================================================================

def search():
    target_int = int(TARGET_HEX, 16)
    
    print("="*70)
    print("EXHAUSTIVE HEADER BYTE-ORDER SEARCH")
    print("="*70)
    print(f"Target (diff=1): {TARGET_HEX[:32]}...")
    print()
    
    # Build coinbase and get merkle root
    coinbase = hex_to_bytes(COINB1_HEX + EXTRANONCE1_HEX + EXTRANONCE2_HEX + COINB2_HEX)
    coinbase_hash = sha256d(coinbase)
    
    print(f"Coinbase hash: {bytes_to_hex(coinbase_hash)}")
    print(f"Expected:      {V7_COINBASE_HASH_HEX}")
    print(f"Match: {bytes_to_hex(coinbase_hash) == V7_COINBASE_HASH_HEX}")
    print()
    
    # Parse integers
    version_int = int(VERSION_BITS_HEX, 16)
    ntime_int = int(NTIME_HEX, 16)
    nbits_int = int(NBITS_HEX, 16)
    nonce_int = int(NONCE_HEX, 16)
    
    # PRE-COMPUTE ALL VARIANTS
    
    # Version options (4 bytes)
    ver_le = struct.pack('<I', version_int)
    ver_be = struct.pack('>I', version_int)
    ver_raw = hex_to_bytes(VERSION_BITS_HEX)
    ver_opts = [("ver_le", ver_le), ("ver_be", ver_be), ("ver_raw", ver_raw)]
    
    # PrevHash options (32 bytes) - this is the medical merkle root
    ph_raw = hex_to_bytes(MEDICAL_MERKLE_ROOT_HEX)  # Direct interpretation
    ph_rev = reverse_bytes(ph_raw)  # Full reverse
    ph_swap = hex_to_bytes(swap_words(MEDICAL_MERKLE_ROOT_HEX))  # 32-bit word swap
    ph_swap_rev = reverse_bytes(ph_swap)  # Swap then reverse
    ph_opts = [("ph_raw", ph_raw), ("ph_rev", ph_rev), ("ph_swap", ph_swap), ("ph_swap_rev", ph_swap_rev)]
    
    # Merkle root options (32 bytes) - the coinbase hash
    mk_raw = coinbase_hash
    mk_rev = reverse_bytes(coinbase_hash)
    mk_opts = [("mk_raw", mk_raw), ("mk_rev", mk_rev)]
    
    # nTime options (4 bytes)
    tm_le = struct.pack('<I', ntime_int)
    tm_be = struct.pack('>I', ntime_int)
    tm_opts = [("tm_le", tm_le), ("tm_be", tm_be)]
    
    # nBits options (4 bytes)
    nb_le = struct.pack('<I', nbits_int)
    nb_be = struct.pack('>I', nbits_int)
    nb_opts = [("nb_le", nb_le), ("nb_be", nb_be)]
    
    # Nonce options (4 bytes)
    nc_le = struct.pack('<I', nonce_int)
    nc_be = struct.pack('>I', nonce_int)
    nc_opts = [("nc_le", nc_le), ("nc_be", nc_be)]
    
    print("Searching...")
    
    count = 0
    for v_name, v_b in ver_opts:
        for p_name, p_b in ph_opts:
            for m_name, m_b in mk_opts:
                for t_name, t_b in tm_opts:
                    for b_name, b_b in nb_opts:
                        for n_name, n_b in nc_opts:
                            count += 1
                            
                            # Build 80-byte header
                            header = v_b + p_b + m_b + t_b + b_b + n_b
                            
                            if len(header) != 80:
                                continue
                            
                            block_hash = sha256d(header)
                            hash_int = int.from_bytes(block_hash, 'little')
                            
                            if hash_int < target_int:
                                print()
                                print("="*70)
                                print(">>> FOUND VALID HEADER! <<<")
                                print("="*70)
                                print(f"Version:  {v_name}")
                                print(f"PrevHash: {p_name}")
                                print(f"Merkle:   {m_name}")
                                print(f"nTime:    {t_name}")
                                print(f"nBits:    {b_name}")
                                print(f"Nonce:    {n_name}")
                                print()
                                print(f"Header: {bytes_to_hex(header)}")
                                print(f"Hash:   {bytes_to_hex(reverse_bytes(block_hash))}")
                                return True
    
    print(f"\nTested {count} combinations. No valid header found.")
    print()
    print("This means either:")
    print("1. The coinbase hash is wrong")
    print("2. The miner is using different extranonces than we captured")
    print("3. The ASIC has a bug or uses non-standard header format")
    return False

if __name__ == "__main__":
    search()
