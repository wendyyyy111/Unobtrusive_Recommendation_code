# src/catalog.py
from typing import Dict


item_catalog = [
    {"id": 0, "name": "Entertainment Content", "sensitivity": 0.1, "category": "A"},
    {"id": 1, "name": "Mild Health Advice", "sensitivity": 0.3, "category": "B"},
    {"id": 2, "name": "Sleep / Schedule Advice", "sensitivity": 0.5, "category": "B"},
    {"id": 3, "name": "Stress Relief Exercises", "sensitivity": 0.7, "category": "C"},
    {"id": 4, "name": "Mental Health Self-Assessment / Information", "sensitivity": 0.9, "category": "C"},
    {"id": 5, "name": "Anxiety Intervention Card (Direct Identification)", "sensitivity": 1.0, "category": "D"},
    {"id": 6, "name": "Work / Study Efficiency Advice", "sensitivity": 0.6, "category": "B"},
    {"id": 7, "name": "Mindfulness / Breathing Meditation Courses", "sensitivity": 0.7, "category": "C"},
    {"id": 8, "name": "Social / Family Interaction Advice", "sensitivity": 0.4, "category": "B"},
    {"id": 9, "name": "Physical Fitness / Exercise Advice", "sensitivity": 0.4, "category": "B"},
    {"id": 10, "name": "News / Information Content", "sensitivity": 0.2, "category": "A"},
    {"id": 11, "name": "Hobby Development Courses", "sensitivity": 0.2, "category": "A"},
]
```


def rel_score(state: int, item_id: int) -> float:
    if state == 1:
        base = {0: 0.8, 1: 0.6, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1,
                6: 0.5, 7: 0.3, 8: 0.4, 9: 0.5, 10: 0.6, 11: 0.6}
    elif state == 2:
        base = {0: 0.3, 1: 0.5, 2: 0.6, 3: 0.9, 4: 0.8, 5: 1.0,
                6: 0.7, 7: 0.8, 8: 0.4, 9: 0.5, 10: 0.3, 11: 0.4}
    elif state == 3:
        base = {0: 1.0, 1: 0.6, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.0,
                6: 0.3, 7: 0.4, 8: 0.7, 9: 0.6, 10: 0.5, 11: 0.8}
    elif state == 4:
        base = {0: 0.4, 1: 0.7, 2: 0.8, 3: 0.5, 4: 0.3, 5: 0.2,
                6: 0.6, 7: 0.9, 8: 0.5, 9: 0.6, 10: 0.3, 11: 0.5}
    else:
        base = {i: 0.0 for i in range(len(item_catalog))}
    return base.get(item_id, 0.0)


def item_sensitivity(item_id: int) -> float:
    return item_catalog[item_id]["sensitivity"]


def signal_privacy(state: int) -> float:
    if state == 2:
        return 1.0
    if state == 4:
        return 0.8
    if state == 3:
        return 0.5
    if state == 1:
        return 0.3
    return 0.2


def comfort_penalty(state: int, item_id: int) -> float:
    if item_id == 5:
        return 1.0 if state == 2 else 0.6
    if item_id == 4:
        return 0.8 if state == 2 else 0.4
    if item_id == 3:
        return 0.4 if state == 2 else 0.2
    return 0.1


def get_item_category_dict():
    return {item["id"]: item["category"] for item in item_catalog}
