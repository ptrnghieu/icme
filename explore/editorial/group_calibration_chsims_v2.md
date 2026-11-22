# Grouping calibration v2 on CH-SIMS (n=2281 clips, 60 true videos)
| features | threshold | #groups | largest group share | ARI | completeness | LEAKAGE (5-fold grouped CV) |
|---|---|---|---|---|---|---|
| spk | 0.10 | 626 | 0.033 | 0.038 | 0.412 | 1.000 |
| spk | 0.15 | 250 | 0.101 | 0.043 | 0.332 | 1.000 |
| spk | 0.20 | 102 | 0.186 | 0.034 | 0.283 | 1.000 |
| spk | 0.25 | 51 | 0.495 | 0.016 | 0.273 | 1.000 |
| spk | 0.30 | 23 | 0.532 | 0.012 | 0.251 | 0.996 |
| spk | 0.35 | 14 | 0.569 | 0.010 | 0.234 | 0.984 |
| spk | 0.40 | 5 | 0.612 | 0.008 | 0.212 | 0.971 |
| spk | 0.45 | 2 | 0.625 | 0.007 | 0.193 | nan |
| spk | 0.50 | 2 | 0.625 | 0.007 | 0.193 | nan |
| spk | 0.55 | 1 | 1.000 | 0.000 | 1.000 | nan |
| spk | 0.60 | 1 | 1.000 | 0.000 | 1.000 | nan |
| spk | 0.70 | 1 | 1.000 | 0.000 | 1.000 | nan |
| clip | 0.10 | 1571 | 0.018 | 0.067 | 0.535 | 1.000 |
| clip | 0.15 | 948 | 0.035 | 0.179 | 0.579 | 1.000 |
| clip | 0.20 | 476 | 0.060 | 0.265 | 0.607 | 1.000 |
| clip | 0.25 | 196 | 0.113 | 0.239 | 0.618 | 0.991 |
| clip | 0.30 | 72 | 0.466 | 0.069 | 0.610 | 0.954 |
| clip | 0.35 | 24 | 0.554 | 0.044 | 0.590 | 0.913 |
| clip | 0.40 | 9 | 0.993 | 0.000 | 0.504 | 0.095 |
| clip | 0.45 | 3 | 0.995 | 0.000 | 0.494 | nan |
| clip | 0.50 | 1 | 1.000 | 0.000 | 1.000 | nan |
| clip | 0.55 | 1 | 1.000 | 0.000 | 1.000 | nan |
| clip | 0.60 | 1 | 1.000 | 0.000 | 1.000 | nan |
| clip | 0.70 | 1 | 1.000 | 0.000 | 1.000 | nan |
| both | 0.10 | 1688 | 0.011 | 0.045 | 0.526 | 1.000 |
| both | 0.15 | 964 | 0.032 | 0.161 | 0.564 | 1.000 |
| both | 0.20 | 417 | 0.062 | 0.225 | 0.563 | 1.000 |
| both | 0.25 | 155 | 0.185 | 0.124 | 0.520 | 1.000 |
| both | 0.30 | 45 | 0.373 | 0.041 | 0.443 | 0.990 |
| both | 0.35 | 16 | 0.599 | 0.011 | 0.281 | 0.964 |
| both | 0.40 | 6 | 0.612 | 0.010 | 0.247 | 0.946 |
| both | 0.45 | 2 | 1.000 | 0.000 | 0.487 | nan |
| both | 0.50 | 1 | 1.000 | 0.000 | 1.000 | nan |
| both | 0.55 | 1 | 1.000 | 0.000 | 1.000 | nan |
| both | 0.60 | 1 | 1.000 | 0.000 | 1.000 | nan |
| both | 0.70 | 1 | 1.000 | 0.000 | 1.000 | nan |

Random 5-fold CV leakage for reference: 1.000
Chosen (min leakage, largest group <= 20%, >= 25 groups): features=clip, threshold=0.25, groups=196, leakage=0.991


## Kết luận (2026-11-22)
- Không có cách gom nhóm tự động nào (giọng người nói, CLIP-B/32, hay cả hai) giảm được rò rỉ nguồn khi vẫn đạt ràng buộc: nhóm lớn nhất ≤ 20% và có ≥ 25 nhóm.
  - Tỉ lệ rò rỉ: 0,95–1,00. Chia ngẫu nhiên: 1,00.
  - Rò rỉ chỉ giảm khi gần như toàn bộ clip dồn vào một nhóm (clip, 0,40: nhóm lớn nhất chiếm 99%).
- Lý do: độ completeness chỉ đạt khoảng 0,6. Mỗi video bị tách ra nhiều cụm, nên thế nào cũng có clip cùng video nằm ở cả train lẫn test.
- **Hệ quả cho E′:** scheme G trong thiết kế đã đăng ký thực chất không khác chia ngẫu nhiên. Vì vậy nó không kiểm soát được kiểu rò rỉ đã giải thích khoảng một nửa tín hiệu ở P6. **E′ chưa được chạy**; không có dữ liệu MER2024 nào bị xem.
