#!/usr/bin/env python3
"""
Debug script V4: Focus on nBits encoding
The miner uses nBits to determine its internal target.
We need to send the CORRECT nBits for our desired difficulty.
"""
import hashlib
import struct

# =============================================================================
# TARGET CALCULATIONS
# =============================================================================

def target_to_nbits_correct(target: int) -> int:
    """
    Convert a target integer to compact nBits format.
    
    nBits format: 
    - High byte = exponent (size in bytes)
    - Low 3 bytes = mantissa (first 3 significant bytes)
    
    Target = mantissa * 2^(8 * (exponent - 3))
    """
    if target == 0:
        return 0
    
    # Convert target to bytes
    target_bytes = target.to_bytes((target.bit_length() + 7) // 8, 'big')
    
    # Size is the number of bytes
    size = len(target_bytes)
    
    # Mantissa is first 3 bytes (or less if target is small)
    if size >= 3:
        mantissa = int.from_bytes(target_bytes[:3], 'big')
    else:
        mantissa = int.from_bytes(target_bytes, 'big') << (8 * (3 - size))
    
    # If high bit of mantissa is set, we need to adjust
    # (because mantissa must be positive in the protocol)
    if mantissa & 0x800000:
        mantissa >>= 8
        size += 1
    
    # nBits = (size << 24) | mantissa
    return (size << 24) | (mantissa & 0x00FFFFFF)

def nbits_to_target(nbits: int) -> int:
    """Convert compact nBits to full target"""
    exp = (nbits >> 24) & 0xFF
    mant = nbits & 0x00FFFFFF
    
    # If high bit of mantissa set, it's negative (we treat as zero)
    if exp <= 3:
        return mant >> (8 * (3 - exp))
    else:
        return mant << (8 * (exp - 3))

def difficulty_to_target(difficulty: float) -> int:
    diff1_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return int(diff1_target / difficulty)

# =============================================================================
# TEST
# =============================================================================

def test():
    print("="*70)
    print("NBITS ENCODING TEST")
    print("="*70)
    
    difficulty = 0.08
    target = difficulty_to_target(difficulty)
    
    print(f"Difficulty: {difficulty}")
    print(f"Target:     {format(target, '064x')}")
    
    # Calculate correct nBits
    nbits = target_to_nbits_correct(target)
    print(f"\nCalculated nBits: {hex(nbits)}")
    
    # Verify by converting back
    target_back = nbits_to_target(nbits)
    print(f"Target from nBits: {format(target_back, '064x')}")
    
    # Compare
    if target == target_back:
        print("EXACT MATCH!")
    else:
        ratio = target / target_back if target_back > 0 else float('inf')
        print(f"Mismatch! Ratio: {ratio:.4f}")
        
    # What V6 was using
    print("\n--- V6 COMPARISON ---")
    v6_nbits = 0x1d0000c7
    v6_target = nbits_to_target(v6_nbits)
    print(f"V6 nBits: {hex(v6_nbits)}")
    print(f"V6 Target: {format(v6_target, '064x')}")
    
    # The ratio between what we wanted and what the miner used
    if v6_target > 0:
        print(f"Ratio (our_target / miner_target): {target / v6_target:.2f}")
    
    # This explains why all shares "fail" verification!
    # The miner is finding hashes below v6_target (which is MUCH HIGHER than our target)
    # But we're checking against our target (which is MUCH LOWER)
    
    print("\n" + "="*70)
    print("SOLUTION:")
    print("="*70)
    print(f"Send nBits: {hex(nbits)} (little-endian in header: {format(nbits, '08x')[::-1]})")
    print("OR")
    print("Accept that miner mines at its own difficulty and just trust shares")

if __name__ == "__main__":
    test()
