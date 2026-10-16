import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.linalg import solve_banded
import hashlib
import time

# ============================================================
# 1. CONFIGURATION
# bird, alaram, female, and male speech ============================================================

AUDIO_PATH = r"C:\Users\sadda\Downloads\male speech.mp3.wav"

KEY = hashlib.sha256(
    b"GraphAudioResearchKey2026"
).digest()

# Fixed nonce for reproducible experiments only.
NONCE = hashlib.sha256(
    b"GraphAudioFixedExperimentNonce"
).digest()[:16]

SESSION_KEY = hashlib.sha256(
    KEY + NONCE
).digest()

MOD = 65536
MASK = 65535


# ============================================================
# 2. READ AUDIO FILE
# ============================================================

sample_rate, audio = wavfile.read(AUDIO_PATH)

if audio.dtype != np.int16:
    raise ValueError("Only 16-bit PCM WAV is supported.")

original_shape = audio.shape

plain = (
    np.ascontiguousarray(audio)
    .reshape(-1)
    .view(np.uint16)
    .copy()
)

n = len(plain)

if n < 10:
    raise ValueError("At least 10 audio samples required.")

indices = np.arange(n, dtype=np.int64)


# ============================================================
# 3. PATH GRAPH AND THREE LANDMARKS
# ============================================================

# Path graph P_n has:
# Vertices = n
# Edges = n - 1

landmarks = (
    n // 5,
    n // 2,
    (4 * n) // 5
)

l1, l2, l3 = landmarks

assert len(set(landmarks)) == 3
assert all(v not in (0, 1) for v in landmarks)

# Shortest path distances from three landmarks

d1 = np.abs(indices - l1)
d2 = np.abs(indices - l2)
d3 = np.abs(indices - l3)


# ============================================================
# 4. COMPUTE SIX CENTRALITY MEASURES
# ============================================================

def compute_centralities(n):

    i = np.arange(n, dtype=np.float64)

    # 1. DEGREE CENTRALITY
    degree = np.full(n, 2.0)
    degree[0] = 1.0
    degree[-1] = 1.0

    # 2. CLOSENESS CENTRALITY
    distance_sum = (
        i * (i + 1) / 2
        + (n - 1 - i) * (n - i) / 2
    )

    closeness = (n - 1) / distance_sum

    # 3. BETWEENNESS CENTRALITY
    betweenness = i * (n - 1 - i)

    # 4. HARMONIC CENTRALITY
    h = np.zeros(n, dtype=np.float64)

    h[1:] = np.cumsum(
        1.0 / np.arange(1, n, dtype=np.float64)
    )

    harmonic = h + h[::-1]

    # 5. EIGENVECTOR CENTRALITY
    eigenvector = np.sin(
        np.pi * (i + 1) / (n + 1)
    )

    # 6. KATZ CENTRALITY
    alpha = 0.2

    ab = np.zeros((3, n), dtype=np.float64)

    ab[0, 1:] = -alpha
    ab[1, :] = 1.0
    ab[2, :-1] = -alpha

    katz = solve_banded(
        (1, 1),
        ab,
        np.ones(n)
    )

    return [
        degree,
        closeness,
        betweenness,
        harmonic,
        eigenvector,
        katz
    ]


centralities = compute_centralities(n)


# ============================================================
# 5. CENTRALITY SELECTION USING THREE LANDMARKS
# ============================================================

# Choose closest landmark.
# np.argmin breaks ties in favor of first landmark.

nearest = np.argmin(
    np.stack((d1, d2, d3), axis=0),
    axis=0
)

# Six non-overlapping conditions

conditions = [
    (nearest == 0) & (indices <= l1),
    (nearest == 0) & (indices > l1),

    (nearest == 1) & (indices <= l2),
    (nearest == 1) & (indices > l2),

    (nearest == 2) & (indices <= l3),
    (nearest == 2) & (indices > l3)
]

names = [
    "Degree",
    "Closeness",
    "Betweenness",
    "Harmonic",
    "Eigenvector",
    "Katz"
]

selected = np.full(n, -1, dtype=np.int8)

for j, condition in enumerate(conditions):
    selected[condition] = j

assert np.all(selected >= 0)

# Normalize centrality values into 16-bit integers

graph_values = np.zeros(n, dtype=np.uint16)

for j, values in enumerate(centralities):

    low = values.min()
    high = values.max()

    if high > low:
        normalized = (values - low) / (high - low)
    else:
        normalized = np.zeros(n, dtype=np.float64)

    quantized = np.rint(
        normalized * MASK
    ).astype(np.uint16)

    graph_values[selected == j] = (
        quantized[selected == j]
    )


# ============================================================
# 6. BIJECTIVE NONLINEAR CHAIN MIXING
# ============================================================

# Separate round keys for all three diffusion passes.

ROUND_KEYS = {}

for label in (b"F1", b"B1", b"F2"):

    digest = hashlib.sha256(
        SESSION_KEY + label
    ).digest()

    ROUND_KEYS[label] = [
        int.from_bytes(digest[j:j+2], "little")
        for j in range(0, 12, 2)
    ]


def chain_mix(previous, index, label):

    k = ROUND_KEYS[label]

    x = int(previous) & MASK
    i = int(index)

    # Step 1: Key-dependent modular addition
    x = (x + k[0] + i) & MASK

    # Step 2: Invertible XOR-shift
    x ^= x >> 7

    # Step 3: Odd multiplication
    x = (x * (k[1] | 1)) & MASK

    # Step 4: Modular addition
    x = (x + k[2]) & MASK

    # Step 5: Nonlinear bit mixing
    x ^= x >> 9

    # Step 6: Odd multiplication
    x = (x * (k[3] | 1)) & MASK

    # Step 7: XOR-shift
    x ^= x >> 5

    # Step 8: Position-dependent addition
    x = (x + k[4] + 3 * i) & MASK

    # Step 9: Odd multiplication
    x = (x * (k[5] | 1)) & MASK

    # Step 10: Final XOR-shift
    x ^= x >> 8

    return x & MASK


# ============================================================
# 7. CHAINED DIFFUSION PASS
# ============================================================

def diffusion_pass(values, keys, label, forward=True):

    size = len(values)

    result = np.empty(
        size, dtype=np.uint16
    )

    previous = 0

    if forward:
        order = range(size)
    else:
        order = range(size - 1, -1, -1)

    for i in order:

        mixing = chain_mix(
            previous, i, label
        )

        result[i] = (
            int(values[i])
            + int(keys[i])
            + mixing
        ) & MASK

        previous = int(result[i])

    return result


# ============================================================
# 8. INVERSE DIFFUSION PASS
# ============================================================

def inverse_pass(values, keys, label, forward=True):

    size = len(values)

    result = np.empty(
        size, dtype=np.uint16
    )

    previous = 0

    if forward:
        order = range(size)
    else:
        order = range(size - 1, -1, -1)

    for i in order:

        mixing = chain_mix(
            previous, i, label
        )

        result[i] = (
            int(values[i])
            - int(keys[i])
            - mixing
        ) & MASK

        # Chaining uses previous ciphertext value
        previous = int(values[i])

    return result


# ============================================================
# 9. THREE-PASS DIFFUSION
# ============================================================

def diffuse(values, graph_keys):

    # Pass 1: Forward
    stage1 = diffusion_pass(
        values,
        graph_keys,
        b"F1",
        forward=True
    )

    # Pass 2: Backward
    stage2 = diffusion_pass(
        stage1,
        graph_keys,
        b"B1",
        forward=False
    )

    # Pass 3: Forward
    stage3 = diffusion_pass(
        stage2,
        graph_keys,
        b"F2",
        forward=True
    )

    return stage3


# ============================================================
# 10. INVERSE THREE-PASS DIFFUSION
# ============================================================

def inverse_diffusion(values, graph_keys):

    # Undo Pass 3
    stage2 = inverse_pass(
        values,
        graph_keys,
        b"F2",
        forward=True
    )

    # Undo Pass 2
    stage1 = inverse_pass(
        stage2,
        graph_keys,
        b"B1",
        forward=False
    )

    # Undo Pass 1
    recovered = inverse_pass(
        stage1,
        graph_keys,
        b"F1",
        forward=True
    )

    return recovered


# ============================================================
# 11. CYCLE GRAPH RESOLVING-SET PERMUTATION
# ============================================================

def cycle_distance(vertices, landmark, size):

    difference = np.abs(vertices - landmark)

    return np.minimum(
        difference,
        size - difference
    )


def generate_permutation(size):

    vertices = np.arange(
        size, dtype=np.int64
    )

    # Resolving set W = {v_0, v_1}
    r1 = cycle_distance(vertices, 0, size)
    r2 = cycle_distance(vertices, 1, size)

    # Sort unique distance representations
    permutation = np.lexsort((r2, r1))

    return permutation


permutation = generate_permutation(n)

assert np.unique(permutation).size == n


# ============================================================
# 12. CONFUSION AND INVERSE CONFUSION
# ============================================================

def confusion(values):
    return values[permutation].copy()


def inverse_confusion(values):

    recovered = np.empty_like(values)

    recovered[permutation] = values

    return recovered


# ============================================================
# 13. ENCRYPTION
# ============================================================

start = time.perf_counter()

diffused = diffuse(plain, graph_values)

encrypted = confusion(diffused)

encryption_time = time.perf_counter() - start


# ============================================================
# 14. DECRYPTION
# ============================================================

start = time.perf_counter()

inverse_permuted = inverse_confusion(encrypted)

decrypted = inverse_diffusion(
    inverse_permuted,
    graph_values
)

decryption_time = time.perf_counter() - start

exact_recovery = np.array_equal(
    plain,
    decrypted
)

assert exact_recovery, "Decryption failed!"


# ============================================================
# 15. CORRELATION ANALYSIS
# ============================================================

def correlation(values):

    x = values.view(np.int16).astype(np.float64)

    return np.corrcoef(
        x[:-1],
        x[1:]
    )[0, 1]


original_corr = correlation(plain)
encrypted_corr = correlation(encrypted)
decrypted_corr = correlation(decrypted)


# ============================================================
# 16. SHANNON ENTROPY
# ============================================================

def entropy(values):

    counts = np.bincount(
        values.astype(np.int32),
        minlength=MOD
    )

    probabilities = (
        counts[counts > 0] / len(values)
    )

    return -np.sum(
        probabilities * np.log2(probabilities)
    )


original_entropy = entropy(plain)
encrypted_entropy = entropy(encrypted)
decrypted_entropy = entropy(decrypted)


# ============================================================
# 17. NPCR AND UACI
# ============================================================

# Modify exactly one plaintext sample bit.
# Keep key, nonce, graph values and permutation identical.

modified_plain = plain.copy()

modified_plain[n // 2] ^= np.uint16(1)

modified_diffused = diffuse(
    modified_plain,
    graph_values
)

modified_encrypted = confusion(
    modified_diffused
)


def npcr(a, b):

    return 100.0 * np.mean(a != b)


def uaci(a, b):

    difference = np.abs(
        a.astype(np.float64)
        - b.astype(np.float64)
    )

    return 100.0 * np.mean(
        difference / MASK
    )


npcr_value = npcr(
    encrypted,
    modified_encrypted
)

uaci_value = uaci(
    encrypted,
    modified_encrypted
)


# ============================================================
# 18. HISTOGRAM CHI-SQUARE ANALYSIS
# ============================================================

def histogram_chi_square(values):

    counts = np.bincount(
        values.astype(np.int32),
        minlength=MOD
    )

    expected = len(values) / MOD

    return np.sum(
        (counts - expected) ** 2 / expected
    )


original_chi = histogram_chi_square(plain)
encrypted_chi = histogram_chi_square(encrypted)
decrypted_chi = histogram_chi_square(decrypted)


# ============================================================
# 19. FINAL OUTPUT - PRINT RESULTS ONLY ONCE
# ============================================================

print("\n" + "=" * 55)
print("       GRAPH-BASED AUDIO ENCRYPTION")
print("=" * 55)

print("\n========== AUDIO INFORMATION ==========")
print(f"Sampling Rate       : {sample_rate}")
print(f"Total Samples       : {n}")
print(f"Duration            : {len(audio)/sample_rate:.4f} s")
print(f"Data Type           : {audio.dtype}")

print("\n========== GRAPH INFORMATION ==========")
print("Diffusion Graph     : Path Graph")
print(f"Vertices            : {n}")
print(f"Edges               : {n-1}")
print(f"Landmarks           : {landmarks}")
print("Confusion Graph     : Cycle Graph")
print("Resolving Set       : (0, 1)")

print("\n========== CENTRALITY SELECTION ==========")

for j, name in enumerate(names):

    count = np.count_nonzero(
        selected == j
    )

    print(f"{name:20s}: {count}")

print("\n========== ENCRYPTION ==========")
print(f"Encryption Time     : {encryption_time:.6f} s")
print(f"Decryption Time     : {decryption_time:.6f} s")
print(f"Exact Recovery      : {exact_recovery}")

print("\n========== CORRELATION ==========")
print(f"Original            : {original_corr:.6f}")
print(f"Encrypted           : {encrypted_corr:.6f}")
print(f"Decrypted           : {decrypted_corr:.6f}")

print("\n========== ENTROPY ==========")
print(f"Original            : {original_entropy:.6f}")
print(f"Encrypted           : {encrypted_entropy:.6f}")
print(f"Decrypted           : {decrypted_entropy:.6f}")

print("\n========== DIFFERENTIAL ANALYSIS ==========")
print(f"NPCR                : {npcr_value:.6f}%")
print(f"UACI                : {uaci_value:.6f}%")

print("\n========== HISTOGRAM ANALYSIS ==========")
print(f"Original Chi-Square : {original_chi:.6f}")
print(f"Encrypted Chi-Square: {encrypted_chi:.6f}")
print(f"Decrypted Chi-Square: {decrypted_chi:.6f}")

print("\n========== VERIFICATION ==========")
print(f"Exact Recovery      : {exact_recovery}")

print("=" * 55)


# ============================================================
# 20. WAVEFORM VISUALIZATION
# ============================================================

original_signed = plain.view(np.int16)
encrypted_signed = encrypted.view(np.int16)
decrypted_signed = decrypted.view(np.int16)

signals = [
    original_signed,
    encrypted_signed,
    decrypted_signed
]

titles = [
    "Original Audio",
    "Encrypted Audio",
    "Decrypted Audio"
]

show_samples = min(n, 5000)

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 8)
)

for ax, signal, title in zip(
    axes, signals, titles
):

    ax.plot(
        signal[:show_samples],
        linewidth=0.6
    )

    ax.set_title(title + " Waveform")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# 21. HISTOGRAM VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 9)
)

for ax, signal, title in zip(
    axes, signals, titles
):

    ax.hist(
        signal,
        bins=256,
        range=(-32768, 32768)
    )

    ax.set_title(title + " Histogram")
    ax.set_xlabel("Sample Value")
    ax.set_ylabel("Frequency")
    ax.set_xlim(-32768, 32767)
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# 21B. SCATTER PLOTS (NEW)
# ============================================================

# A random subset of points is plotted so the figures stay
# fast and readable for long recordings. The seed is fixed
# so the same points are chosen on every run.

scatter_rng = np.random.default_rng(12345)

num_points = min(n - 1, 10000)

pick = np.sort(
    scatter_rng.choice(
        n - 1,
        size=num_points,
        replace=False
    )
)

# Normalize 16-bit samples to the range [-1, 1) so the plot axes
# match the scale of the correlation coefficient.
# Scaling does not change the correlation values.
orig_norm = original_signed.astype(np.float64) / 32768.0
enc_norm = encrypted_signed.astype(np.float64) / 32768.0

# ---- Figure A: adjacent-sample scatter (x[i] vs x[i+1]) ----
# Original audio: points cluster along the diagonal (high correlation).
# Encrypted audio: points should fill the whole square uniformly.

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 6)
)

for ax, values, title, corr in zip(
    axes,
    (orig_norm, enc_norm),
    ("Original Audio", "Encrypted Audio"),
    (original_corr, encrypted_corr)
):

    ax.scatter(
        values[pick],
        values[pick + 1],
        s=2,
        alpha=0.35
    )

    ax.set_title(
        f"{title}\n"
        f"Correlation = {corr:.6f}"
    )
    ax.set_xlabel("Normalized sample value x[i]")
    ax.set_ylabel("Normalized sample value x[i+1]")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ---- Figure B: original vs encrypted scatter at the same index ----
# Each point is (original sample i, encrypted sample i).
# For a good cipher this should look like uniform noise with no
# visible relationship between plaintext and ciphertext.

pick_b = np.sort(
    scatter_rng.choice(
        n,
        size=min(n, 10000),
        replace=False
    )
)

plain_vs_cipher_corr = np.corrcoef(
    orig_norm.astype(np.float64),
    enc_norm.astype(np.float64)
)[0, 1]

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    orig_norm[pick_b],
    enc_norm[pick_b],
    s=2,
    alpha=0.35
)

ax.set_title(
    "Original vs Encrypted Samples\n"
    f"Correlation = {plain_vs_cipher_corr:.6f}"
)
ax.set_xlabel("Original sample value (normalized)")
ax.set_ylabel("Encrypted sample value (normalized)")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()

print("\n========== SCATTER ANALYSIS ==========")
print(f"Points plotted                     : {num_points}")
print(f"Adjacent corr. (original)          : {original_corr:.6f}")
print(f"Adjacent corr. (encrypted)         : {encrypted_corr:.6f}")
print(f"Original vs encrypted correlation  : {plain_vs_cipher_corr:.6f}")


# ============================================================
# 22. CENTRALITY VISUALIZATION
# ============================================================

display_n = min(n, 3000)

fig, axes = plt.subplots(
    2, 1,
    figsize=(12, 7)
)

axes[0].plot(
    graph_values[:display_n],
    linewidth=0.8
)

axes[0].set_title("Graph-Derived Vertex Values")
axes[0].set_xlabel("Vertex Index")
axes[0].set_ylabel("16-bit Value")
axes[0].grid(alpha=0.25)

axes[1].scatter(
    np.arange(display_n),
    selected[:display_n],
    s=3
)

axes[1].set_title("Centrality Selection")
axes[1].set_xlabel("Vertex Index")
axes[1].set_ylabel("Centrality Measure")
axes[1].set_yticks(range(6))
axes[1].set_yticklabels(names)
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ============================================================
# 23. RECOVER ORIGINAL AUDIO IN MEMORY
# ============================================================

recovered_audio = (
    decrypted.view(np.int16)
    .reshape(original_shape)
)

# ============================================================
# 24. KEY SENSITIVITY ANALYSIS (NEW)
# ============================================================

# Each trial flips ONE bit of the 256-bit key, encrypts the same
# plaintext again and compares the two ciphertexts.
# Each trial costs one extra full encryption (3 diffusion passes),
# so increase this number only if you can afford the run time.
KEY_SENS_TRIALS = 5


def derive_round_keys(key_bytes):

    session = hashlib.sha256(
        key_bytes + NONCE
    ).digest()

    keys = {}

    for label in (b"F1", b"B1", b"F2"):

        digest = hashlib.sha256(
            session + label
        ).digest()

        keys[label] = [
            int.from_bytes(digest[j:j+2], "little")
            for j in range(0, 12, 2)
        ]

    return keys


def set_round_keys(new_keys):

    # chain_mix reads the global ROUND_KEYS dictionary,
    # so it is updated in place.
    ROUND_KEYS.clear()
    ROUND_KEYS.update(new_keys)


ORIGINAL_ROUND_KEYS = {
    label: list(values)
    for label, values in ROUND_KEYS.items()
}

# Sanity check: the helper reproduces the original key schedule
assert derive_round_keys(KEY) == ORIGINAL_ROUND_KEYS


def flip_key_bit(key_bytes, bit_position):

    modified = bytearray(key_bytes)
    modified[bit_position // 8] ^= 1 << (bit_position % 8)

    return bytes(modified)


def bit_difference(a, b):

    # Percentage of differing bits between two uint16 arrays
    xor_bytes = (a ^ b).view(np.uint8)

    return 100.0 * np.unpackbits(xor_bytes).mean()


# ---------- 24.1 Ciphertext change after a 1-bit key change ----------

key_rng = np.random.default_rng(2026)

flipped_bits = key_rng.choice(
    256,
    size=KEY_SENS_TRIALS,
    replace=False
)

ks_npcr = []
ks_uaci = []
ks_bits = []

first_modified_cipher = None

print("\n========== KEY SENSITIVITY ==========")
print(f"Trials (one flipped key bit each) : {KEY_SENS_TRIALS}")

for t, bit in enumerate(flipped_bits):

    bit = int(bit)

    print(
        f"Running trial {t + 1}/{KEY_SENS_TRIALS} "
        f"(flipping key bit {bit}) ...",
        flush=True
    )

    set_round_keys(
        derive_round_keys(flip_key_bit(KEY, bit))
    )

    modified_cipher = confusion(
        diffuse(plain, graph_values)
    )

    set_round_keys(ORIGINAL_ROUND_KEYS)

    if first_modified_cipher is None:
        first_modified_cipher = modified_cipher

    ks_npcr.append(npcr(encrypted, modified_cipher))
    ks_uaci.append(uaci(encrypted, modified_cipher))
    ks_bits.append(bit_difference(encrypted, modified_cipher))

    print(
        f"  Key bit {bit:3d} | "
        f"NPCR = {ks_npcr[-1]:.6f}% | "
        f"UACI = {ks_uaci[-1]:.6f}% | "
        f"Bit difference = {ks_bits[-1]:.4f}%"
    )

print("\n---------- Key sensitivity summary (mean +/- std) ----------")
print(
    f"NPCR           : {np.mean(ks_npcr):.6f}% "
    f"+/- {np.std(ks_npcr):.6f}   (ideal about 99.9985%)"
)
print(
    f"UACI           : {np.mean(ks_uaci):.6f}% "
    f"+/- {np.std(ks_uaci):.6f}   (ideal about 33.334%)"
)
print(
    f"Bit difference : {np.mean(ks_bits):.4f}% "
    f"+/- {np.std(ks_bits):.4f}   (ideal about 50%)"
)


# ---------- 24.2 Decryption with a slightly wrong key ----------

# Uses the first modified key (one flipped bit).

wrong_bit = int(flipped_bits[0])

set_round_keys(
    derive_round_keys(flip_key_bit(KEY, wrong_bit))
)

wrong_decrypted = inverse_diffusion(
    inverse_confusion(encrypted),
    graph_values
)

set_round_keys(ORIGINAL_ROUND_KEYS)

# The original key schedule must be fully restored
assert ROUND_KEYS == ORIGINAL_ROUND_KEYS

assert not np.array_equal(plain, wrong_decrypted)

wrong_match = 100.0 * np.mean(plain == wrong_decrypted)
wrong_bit_diff = bit_difference(plain, wrong_decrypted)
wrong_adj_corr = correlation(wrong_decrypted)

wrong_signed = wrong_decrypted.view(np.int16)

plain_vs_wrong_corr = np.corrcoef(
    original_signed.astype(np.float64),
    wrong_signed.astype(np.float64)
)[0, 1]

print("\n---------- Decryption with one wrong key bit ----------")
print(f"Flipped key bit                    : {wrong_bit}")
print(f"Samples matching original          : {wrong_match:.6f}%")
print(f"Bit difference from original       : {wrong_bit_diff:.4f}%")
print(f"Adjacent correlation (wrong key)   : {wrong_adj_corr:.6f}")
print(f"Original vs wrong-key correlation  : {plain_vs_wrong_corr:.6f}")


# ---------- 24.3 Key sensitivity waveforms (paper style) ----------

# Three stacked panels: amplitude normalized to [-1, 1],
# horizontal axis in seconds, full recording plotted.

channels = audio.shape[1] if audio.ndim > 1 else 1

time_axis = np.arange(n) / (sample_rate * channels)

original_wave = original_signed.astype(np.float64) / 32768.0

correct_wave = (
    decrypted.view(np.int16).astype(np.float64) / 32768.0
)

wrong_wave = wrong_signed.astype(np.float64) / 32768.0

# (signal, panel title, line colour) - change the colours here
wave_panels = [
    (original_wave, "(a) Original Audio", "blue"),
    (correct_wave, "(b) Correct Key Decryption", "green"),
    (wrong_wave, "(c) Wrong Key Decryption", "red")
]

# Panels (a) and (b) use a tight symmetric range around the signal;
# panel (c) uses the full range -1 to 1.
tight_limit = max(
    1.1 * np.max(np.abs(original_wave)),
    1e-3
)

fig, axes = plt.subplots(
    3, 1,
    figsize=(11, 8.5)
)

for k, (ax, (wave, title, color)) in enumerate(
    zip(axes, wave_panels)
):

    ax.plot(
        time_axis,
        wave,
        color=color,
        linewidth=0.5
    )

    ax.set_title(
        title,
        fontsize=10,
        fontweight="bold"
    )

    ax.set_xlim(0, time_axis[-1])

    if k == 2:
        ax.set_ylim(-1, 1)
    else:
        ax.set_ylim(-tight_limit, tight_limit)

axes[-1].set_xlabel(
    "Time (s)",
    fontsize=10,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# ---------- 24.4 Key sensitivity scatter plots ----------

cipher_1 = encrypted_signed.astype(np.float64) / 32768.0

cipher_2 = (
    first_modified_cipher.view(np.int16)
    .astype(np.float64) / 32768.0
)

wrong_norm = wrong_signed.astype(np.float64) / 32768.0

cipher_vs_cipher_corr = np.corrcoef(
    cipher_1,
    cipher_2
)[0, 1]

pick_k = np.sort(
    key_rng.choice(
        n,
        size=min(n, 10000),
        replace=False
    )
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 6)
)

# Left: ciphertext (correct key) vs ciphertext (1-bit-changed key)
axes[0].scatter(
    cipher_1[pick_k],
    cipher_2[pick_k],
    s=2,
    alpha=0.35
)

axes[0].set_title(
    "Ciphertext: Correct Key vs One-Bit-Changed Key\n"
    f"Correlation = {cipher_vs_cipher_corr:.6f}"
)
axes[0].set_xlabel("Encrypted sample (correct key)")
axes[0].set_ylabel("Encrypted sample (changed key)")

# Right: adjacent samples of the wrong-key decryption
axes[1].scatter(
    wrong_norm[pick],
    wrong_norm[pick + 1],
    s=2,
    alpha=0.35
)

axes[1].set_title(
    "Wrong-Key Decrypted Audio\n"
    f"Correlation = {wrong_adj_corr:.6f}"
)
axes[1].set_xlabel("Normalized sample value x[i]")
axes[1].set_ylabel("Normalized sample value x[i+1]")

for ax in axes:
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# No WAV files, images or results are saved.