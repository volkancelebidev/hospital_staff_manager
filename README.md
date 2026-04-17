# Hospital Staff Manager

A hospital staff management system that tracks doctors and nurses
across departments, computes salaries, analyses patient load,
and generates structured workforce reports.

Built to consolidate Python fundamentals, OOP, and NumPy in a single
healthcare-focused project.

---

## Features

- **Doctor & Nurse Management** — registration with duplicate prevention
- **Salary Calculation** — experience-based formula with shift bonuses for nurses
- **Experience Classification** — automatic Senior / Mid / Junior labelling via @property
- **Patient Load Analysis** — Heavy / Moderate / Light classification with NumPy boolean masking
- **Department Headcount** — staff distribution across all departments
- **Shift Distribution** — nurse count and salary cost per shift
- **Top Earners** — highest-paid staff ranked with sorted() and lambda
- **Flexible Queries** — filter by department, shift, or seniority level

---

## Tech Stack

| Layer      | Technology                                      |
|------------|-------------------------------------------------|
| Language   | Python 3.12                                     |
| Paradigm   | OOP — Inheritance, @property, @classmethod      |
| Pattern    | Repository Pattern                              |
| Analytics  | NumPy 2.4.4                                     |

---

## Project Structure
```
hospital-staff-manager/
├── hospital_staff_manager.py    # All domain models + analytics + reporting
└── .gitignore
```
---

## How to Run

```bash
git clone https://github.com/volkancelebidev/hospital_staff_manager.git
cd hospital_staff_manager
pip install numpy
python hospital_staff_manager.py
```

---

## What I Learned

- Designing a class hierarchy with inheritance and super()
- Using @property for computed attributes that stay in sync automatically
- Applying @classmethod as a factory method to build objects from dicts
- Combining list comprehensions with NumPy for analytics
- Implementing the Repository Pattern to centralise data access
