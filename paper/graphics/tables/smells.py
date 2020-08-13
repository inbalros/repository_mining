import pandas as pd

from paper.utils import pandas_to_latex

designite = ["Imperative Abstraction",
             "Multifaceted Abstraction",
             "Unnecessary Abstraction",
             "Unutilized Abstraction",
             "Deficient Encapsulation",
             "Unexploited Encapsulation",
             "Broken Modularization",
             "Cyclic - Dependent Modularization",
             "Insufficient Modularization",
             "Hub - like Modularization",
             "Broken Hierarchy",
             "Cyclic Hierarchy",
             "Deep Hierarchy",
             "Missing Hierarchy",
             "Multipath Hierarchy",
             "Rebellious Hierarchy",
             "Wide Hierarchy"]

fowler = [
    "God Class",
    "Class Data Should Be Private",
    "Complex Class",
    "Lazy Class",
    "Refused Bequest",
    "Spaghetti Code",
    "Speculative Generality",
    "Data Class",
    "Brain Class",
    "Large Class",
    "Swiss Army Knife",
    "Anti Singleton",
    "Feature Envy",
    "Long Method",
    "Long Parameter List",
    "Message Chain",
    "Dispersed Coupling",
    "Intensive Coupling",
    "Shotgun Surgery",
    "Brain Method"
]

df = pd.DataFrame(designite, columns=['Designite Smells'])
pandas_to_latex(df, "designite_smells.tex", vertical_bars=True, right_align_first_column=False, index=False)
df = pd.DataFrame(fowler, columns=['Traditional Smells'])
pandas_to_latex(df, "traditional_smells.tex", vertical_bars=True, right_align_first_column=False, index=False)
