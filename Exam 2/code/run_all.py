"""
run_all.py
Script maestro para reproducir todos los cálculos numéricos y las 7 figuras
del informe del Examen 2 (Astrofísica Galáctica y Extragaláctica).
"""

from part1_analytical import run_part1
from part2_kuzmin_orbits import run_part2
from part3_disk_halo import run_part3


def main():
    run_part1()
    print()
    _, orbit_disk = run_part2()
    print()
    run_part3(part2_orbit=orbit_disk)
    print("\nTodas las simulaciones y figuras se generaron correctamente.")


if __name__ == "__main__":
    main()
