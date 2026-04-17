"""
hospital_staff_manager.py

A hospital staff management system that tracks doctors and nurses
across departments, computes salaries, analyses patient load,
and generates structured workforce reports.

Covered topics:
    Python Fundamentals → f-string, loops, list comprehension, dict, sorted/lambda
    OOP                 → inheritance, super(), @property, @classmethod, __repr__
    NumPy               → array creation, vectorized stats, boolean masking

Typical usage:
    python hospital_staff_manager.py
"""

import numpy as np


# =============================================================================
# DOMAIN MODELS — OOP CLASS HIERARCHY
# =============================================================================


class StaffMember:
    """Base class for all hospital staff types.

    Provides shared attributes (id, name, department, experience) and
    computed properties that subclasses inherit without re-implementing.
    Subclasses must override base_salary.
    """

    # Valid department names — class-level constant shared by all instances.
    DEPARTMENTS = ("Cardiology", "Neurology", "Emergency", "Pediatrics", "Oncology")

    def __init__(
        self,
        staff_id: str,
        name: str,
        department: str,
        years_exp: int,
    ) -> None:
        self.staff_id   = staff_id    # unique identifier — acts as PRIMARY KEY
        self.name       = name
        self.department = department
        self.years_exp  = years_exp   # years of professional experience

    @property
    def experience_level(self) -> str:
        """Seniority tier derived from years of experience.

        Computed on every access so the label stays in sync if years_exp
        is updated — no manual recalculation needed by the caller.

        Returns:
            "Senior", "Mid", or "Junior".
        """
        if self.years_exp >= 10:
            return "Senior"
        if self.years_exp >= 5:
            return "Mid"
        return "Junior"

    @property
    def base_salary(self) -> float:
        """Base monthly salary in USD.

        Subclasses must override this — each role has its own formula.
        Raising NotImplementedError enforces the contract at runtime.
        """
        raise NotImplementedError("Subclass must implement base_salary.")

    def __repr__(self) -> str:
        # __repr__ → readable output when print(staff) is called
        return (
            f"{self.__class__.__name__}("
            f"id={self.staff_id!r}, name={self.name!r}, "
            f"dept={self.department!r}, exp={self.years_exp}yrs)"
        )


class Doctor(StaffMember):
    """A physician registered in the hospital.

    Inherits shared fields from StaffMember and adds specialty and
    weekly patient volume, which drive load classification.
    """

    SPECIALTIES = (
        "Cardiologist", "Neurologist", "Surgeon",
        "Pediatrician", "Oncologist",
    )

    def __init__(
        self,
        staff_id: str,
        name: str,
        department: str,
        years_exp: int,
        specialty: str,
        weekly_patients: int,
    ) -> None:
        # super().__init__() → delegate shared fields to the parent class
        super().__init__(staff_id, name, department, years_exp)
        self.specialty       = specialty
        self.weekly_patients = weekly_patients

    @property
    def base_salary(self) -> float:
        """Doctor salary: fixed base + experience multiplier.

        Formula: 5000 + (years_exp * 1500)
        Computed as a property so it updates automatically when years_exp changes.
        """
        return 5000 + (self.years_exp * 1500)

    @property
    def patient_load(self) -> str:
        """Workload classification based on weekly patient volume.

        Returns:
            "Heavy" (>= 50), "Moderate" (>= 30), or "Light".
        """
        if self.weekly_patients >= 50:
            return "Heavy"
        if self.weekly_patients >= 30:
            return "Moderate"
        return "Light"

    @classmethod
    def from_dict(cls, data: dict) -> "Doctor":
        """Construct a Doctor instance from a plain dictionary.

        @classmethod → called on the class, not an instance: Doctor.from_dict(d)
        Useful when loading staff data from JSON, a database row, or an API.

        Args:
            data: Dict with keys matching Doctor's constructor parameters.

        Returns:
            A fully initialised Doctor instance.
        """
        return cls(
            staff_id        = data["staff_id"],
            name            = data["name"],
            department      = data["department"],
            years_exp       = data["years_exp"],
            specialty       = data["specialty"],
            weekly_patients = data["weekly_patients"],
        )


class Nurse(StaffMember):
    """A nurse assigned to a department and shift.

    Inherits shared fields from StaffMember and adds shift assignment
    and certification count, both of which affect total compensation.
    """

    SHIFTS = ("Morning", "Evening", "Night")

    def __init__(
        self,
        staff_id: str,
        name: str,
        department: str,
        years_exp: int,
        shift: str,
        certifications: int,
    ) -> None:
        super().__init__(staff_id, name, department, years_exp)
        self.shift          = shift
        self.certifications = certifications

    @property
    def base_salary(self) -> float:
        """Nurse base salary: fixed rate + experience + certification bonus.

        Formula: 3000 + (years_exp * 800) + (certifications * 200)
        """
        return 3000 + (self.years_exp * 800) + (self.certifications * 200)

    @property
    def shift_bonus(self) -> float:
        """Additional compensation for unsociable shift hours.

        Dict lookup replaces an if/elif chain — cleaner and easier to extend.

        Returns:
            0 for Morning, 200 for Evening, 500 for Night.
        """
        bonuses = {"Morning": 0.0, "Evening": 200.0, "Night": 500.0}
        return bonuses.get(self.shift, 0.0)

    @property
    def total_salary(self) -> float:
        """Total monthly compensation including shift bonus."""
        return self.base_salary + self.shift_bonus


# =============================================================================
# REPOSITORY — DATA ACCESS LAYER
# =============================================================================


class HospitalStaffManager:
    """Central manager for all hospital staff data.

    Implements the Repository pattern: all add, query, and analytics
    operations are encapsulated here so callers never touch the raw lists.

    Args:
        hospital_name: Display name shown in reports.
    """

    def __init__(self, hospital_name: str) -> None:
        self.hospital_name = hospital_name
        self._doctors: list[Doctor] = []   # _ prefix → internal use only
        self._nurses:  list[Nurse]  = []

    # -------------------------------------------------------------------------
    # Write operations
    # -------------------------------------------------------------------------

    def add_doctor(self, doctor: Doctor) -> None:
        """Register a doctor, skipping duplicates silently.

        Args:
            doctor: A Doctor instance to persist.
        """
        # list comprehension → extract existing IDs into a list
        existing_ids = [d.staff_id for d in self._doctors]
        if doctor.staff_id in existing_ids:
            print(f"[WARN] Doctor {doctor.staff_id} already registered.")
            return
        self._doctors.append(doctor)

    def add_nurse(self, nurse: Nurse) -> None:
        """Register a nurse, skipping duplicates silently.

        Args:
            nurse: A Nurse instance to persist.
        """
        existing_ids = [n.staff_id for n in self._nurses]
        if nurse.staff_id in existing_ids:
            print(f"[WARN] Nurse {nurse.staff_id} already registered.")
            return
        self._nurses.append(nurse)

    # -------------------------------------------------------------------------
    # Query operations
    # -------------------------------------------------------------------------

    def get_doctors_by_department(self, department: str) -> list[Doctor]:
        """Return all doctors assigned to a given department.

        List comprehension with a condition — equivalent to SQL WHERE.

        Args:
            department: One of StaffMember.DEPARTMENTS.

        Returns:
            Filtered list of Doctor instances.
        """
        # for d in self._doctors → assign each doctor to d
        # if d.department == department → keep only matching ones
        return [d for d in self._doctors if d.department == department]

    def get_nurses_by_shift(self, shift: str) -> list[Nurse]:
        """Return all nurses assigned to a given shift.

        Args:
            shift: One of Nurse.SHIFTS.

        Returns:
            Filtered list of Nurse instances.
        """
        return [n for n in self._nurses if n.shift == shift]

    def get_senior_staff(self) -> list[StaffMember]:
        """Return all staff members with Senior experience level.

        Combines doctors and nurses into one list before filtering.
        experience_level is a @property — no separate call needed.

        Returns:
            List of StaffMember instances (mixed Doctor and Nurse).
        """
        all_staff = self._doctors + self._nurses   # merge two lists
        # experience_level @property → computed automatically per object
        return [s for s in all_staff if s.experience_level == "Senior"]

    # -------------------------------------------------------------------------
    # NumPy analytics
    # -------------------------------------------------------------------------

    def doctor_salary_analysis(self) -> dict:
        """Compute descriptive statistics for doctor base salaries.

        List comprehension extracts salaries → np.array() converts to NumPy
        so vectorized statistical functions can run without a loop.

        Returns:
            Dict with mean, median, std, min, max.
        """
        if not self._doctors:
            return {}

        # list comprehension → collect each doctor's salary
        salaries = np.array([d.base_salary for d in self._doctors])

        return {
            "mean"  : round(float(np.mean(salaries)),   2),
            "median": round(float(np.median(salaries)), 2),
            "std"   : round(float(np.std(salaries)),    2),
            "min"   : round(float(np.min(salaries)),    2),
            "max"   : round(float(np.max(salaries)),    2),
        }

    def patient_load_analysis(self) -> dict:
        """Analyse weekly patient volume distribution across doctors.

        Boolean masking counts doctors in each load tier without looping.
        The & operator applies element-wise AND across two boolean arrays.

        Returns:
            Dict with total, average, and per-tier counts.
        """
        if not self._doctors:
            return {}

        # np.array → convert patient counts to a NumPy array
        patients = np.array([d.weekly_patients for d in self._doctors])

        # boolean masking → np.sum() counts True values
        heavy    = np.sum(patients >= 50)
        moderate = np.sum((patients >= 30) & (patients < 50))  # & → AND
        light    = np.sum(patients < 30)

        return {
            "total_weekly_patients": int(np.sum(patients)),
            "avg_per_doctor"       : round(float(np.mean(patients)), 1),
            "heavy_load"           : int(heavy),
            "moderate_load"        : int(moderate),
            "light_load"           : int(light),
        }

    def department_headcount(self) -> dict:
        """Count doctors and nurses per department.

        Returns:
            Nested dict: {department: {doctors, nurses, total}}.
        """
        headcount = {}

        # StaffMember.DEPARTMENTS → class-level constant, accessed on the class
        for dept in StaffMember.DEPARTMENTS:
            # list comprehension → count matching staff per department
            doctor_count = len([d for d in self._doctors if d.department == dept])
            nurse_count  = len([n for n in self._nurses  if n.department == dept])
            headcount[dept] = {
                "doctors": doctor_count,
                "nurses" : nurse_count,
                "total"  : doctor_count + nurse_count,
            }

        return headcount

    def nurse_shift_distribution(self) -> dict:
        """Compute headcount and salary cost per shift.

        np.sum() aggregates total salary cost without an explicit loop.

        Returns:
            Dict: {shift: {count, avg_salary, total_cost}}.
        """
        distribution = {}

        for shift in Nurse.SHIFTS:
            # list comprehension → nurses on this shift
            shift_nurses = [n for n in self._nurses if n.shift == shift]

            if not shift_nurses:
                distribution[shift] = {"count": 0, "avg_salary": 0, "total_cost": 0}
                continue

            # np.array → convert salaries for vectorized aggregation
            salaries = np.array([n.total_salary for n in shift_nurses])

            distribution[shift] = {
                "count"     : len(shift_nurses),
                "avg_salary": round(float(np.mean(salaries)), 2),
                "total_cost": round(float(np.sum(salaries)),  2),
            }

        return distribution

    def top_earners(self, n: int = 3) -> list[StaffMember]:
        """Return the n highest-paid staff members across all roles.

        sorted() with a lambda key avoids defining a named helper function.
        reverse=True → descending order (highest salary first).
        Slice [:n] returns only the first n elements.

        Args:
            n: Number of top earners to return. Defaults to 3.

        Returns:
            List of StaffMember instances ordered by base_salary descending.
        """
        all_staff = self._doctors + self._nurses

        # lambda s: s.base_salary → extract salary from each staff member for comparison
        sorted_staff = sorted(all_staff, key=lambda s: s.base_salary, reverse=True)

        return sorted_staff[:n]   # slice → first n elements

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------

    def print_staff_report(self) -> None:
        """Print a formatted table of all registered doctors and nurses."""

        separator = "=" * 60
        print(f"\n{separator}")
        print(f"  {self.hospital_name} — STAFF REPORT")
        print(separator)
        print(f"  Total Doctors : {len(self._doctors)}")
        print(f"  Total Nurses  : {len(self._nurses)}")
        print(f"  Total Staff   : {len(self._doctors) + len(self._nurses)}")
        print(separator)

        # --- Doctors ---
        print("\n[Doctors]")
        sep = "-" * 70
        print(sep)
        print(f"  {'ID':<6} {'Name':<20} {'Dept':<14} {'Specialty':<15} "
              f"{'Exp':>4} {'Level':<8} {'Patients':>8} {'Load':>8}")
        print(sep)

        # for d in self._doctors → assign each doctor to d
        for d in self._doctors:
            print(
                f"  {d.staff_id:<6} "
                f"{d.name:<20} "
                f"{d.department:<14} "
                f"{d.specialty:<15} "
                f"{d.years_exp:>4} "
                f"{d.experience_level:<8} "
                f"{d.weekly_patients:>8} "
                f"{d.patient_load:>8}"
            )
        print(sep)

        # --- Nurses ---
        print("\n[Nurses]")
        print(sep)
        print(f"  {'ID':<6} {'Name':<20} {'Dept':<14} {'Shift':<10} "
              f"{'Exp':>4} {'Level':<8} {'Salary':>10}")
        print(sep)

        # for n in self._nurses → assign each nurse to n
        for n in self._nurses:
            print(
                f"  {n.staff_id:<6} "
                f"{n.name:<20} "
                f"{n.department:<14} "
                f"{n.shift:<10} "
                f"{n.years_exp:>4} "
                f"{n.experience_level:<8} "
                f"{n.total_salary:>10.0f}"
            )
        print(sep)

    def print_analytics_report(self) -> None:
        """Print NumPy-powered analytics: salaries, load, headcount, shifts."""

        sep = "-" * 46

        print("\n[Doctor Salary Analysis]")
        print(sep)
        # dict.items() → iterate key-value pairs together
        for key, val in self.doctor_salary_analysis().items():
            print(f"  {key:<8}: ${val:,.2f}")
        print(sep)

        print("\n[Patient Load Analysis]")
        print(sep)
        for key, val in self.patient_load_analysis().items():
            print(f"  {key:<24}: {val}")
        print(sep)

        print("\n[Department Headcount]")
        print(sep)
        print(f"  {'Department':<14} {'Doctors':>8} {'Nurses':>8} {'Total':>8}")
        print(sep)
        # for dept, counts → unpack dict items into dept and counts
        for dept, counts in self.department_headcount().items():
            print(
                f"  {dept:<14} "
                f"{counts['doctors']:>8} "
                f"{counts['nurses']:>8} "
                f"{counts['total']:>8}"
            )
        print(sep)

        print("\n[Nurse Shift Distribution]")
        print(sep)
        for shift, data in self.nurse_shift_distribution().items():
            print(
                f"  {shift:<10} | "
                f"Count: {data['count']:>2} | "
                f"Avg: ${data['avg_salary']:>8,.0f} | "
                f"Total: ${data['total_cost']:>9,.0f}"
            )
        print(sep)

        print("\n[Top 3 Earners]")
        print(sep)
        # enumerate(iterable, start=1) → index + element, counting from 1
        for i, staff in enumerate(self.top_earners(3), start=1):
            print(f"  {i}. {staff.name:<20} | ${staff.base_salary:>10,.0f}")
        print(sep)


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """Seed the hospital with sample staff and run all reports."""

    print("[+] Hospital Staff Manager — starting\n")

    manager = HospitalStaffManager("Istanbul Medical Center")

    # -------------------------------------------------------------------------
    # Add doctors via @classmethod factory
    # -------------------------------------------------------------------------
    doctor_records = [
        {"staff_id": "D001", "name": "Dr. Ayse Kara",     "department": "Cardiology",  "years_exp": 15, "specialty": "Cardiologist",  "weekly_patients": 55},
        {"staff_id": "D002", "name": "Dr. Mehmet Yildiz", "department": "Neurology",   "years_exp":  8, "specialty": "Neurologist",   "weekly_patients": 35},
        {"staff_id": "D003", "name": "Dr. Elif Sahin",    "department": "Emergency",   "years_exp":  3, "specialty": "Surgeon",       "weekly_patients": 60},
        {"staff_id": "D004", "name": "Dr. Can Ozturk",    "department": "Pediatrics",  "years_exp": 12, "specialty": "Pediatrician",  "weekly_patients": 45},
        {"staff_id": "D005", "name": "Dr. Zeynep Demir",  "department": "Oncology",    "years_exp":  6, "specialty": "Oncologist",    "weekly_patients": 28},
    ]

    # for data in doctor_records → assign each dict to data
    for data in doctor_records:
        # Doctor.from_dict() → @classmethod builds the object from a dict
        manager.add_doctor(Doctor.from_dict(data))

    # -------------------------------------------------------------------------
    # Add nurses directly
    # -------------------------------------------------------------------------
    nurses = [
        Nurse("N001", "Fatma Celik",  "Cardiology", 7,  "Morning", 3),
        Nurse("N002", "Ali Yurt",     "Neurology",  2,  "Night",   1),
        Nurse("N003", "Selin Arslan", "Emergency",  10, "Evening", 4),
        Nurse("N004", "Burak Gunes",  "Pediatrics", 5,  "Morning", 2),
        Nurse("N005", "Meral Toprak", "Oncology",   8,  "Night",   5),
        Nurse("N006", "Deniz Kaya",   "Emergency",  3,  "Evening", 2),
    ]

    # for nurse in nurses → assign each Nurse object to nurse
    for nurse in nurses:
        manager.add_nurse(nurse)

    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------
    manager.print_staff_report()
    manager.print_analytics_report()

    # -------------------------------------------------------------------------
    # Query examples
    # -------------------------------------------------------------------------
    print("\n[Query Examples]")

    cardio = manager.get_doctors_by_department("Cardiology")
    print(f"\n  Cardiology Doctors ({len(cardio)}):")
    for d in cardio:   # for d in cardio → assign each doctor to d
        print(f"    {d.name} — {d.specialty} — {d.patient_load} load")

    night = manager.get_nurses_by_shift("Night")
    print(f"\n  Night Shift Nurses ({len(night)}):")
    for n in night:    # for n in night → assign each nurse to n
        print(f"    {n.name} — ${n.total_salary:,.0f} (inc. night bonus)")

    seniors = manager.get_senior_staff()
    print(f"\n  Senior Staff ({len(seniors)}):")
    for s in seniors:  # for s in seniors → assign each staff member to s
        print(f"    {s.name} — {s.years_exp} yrs — {s.__class__.__name__}")


if __name__ == "__main__":
    # Runs only when executed directly — not when imported as a module.
    main()