# PROJECT_PYPY
פרויקט ובו אנו נשחזר ניסוי במאמר# # פרויקט שחזור תוצאות מדעיות: התגברות על שכחה קטסטרופלית ברשתות נוירונים

מאגר זה מכיל מחקר שחזור (Reproducibility Study) למאמר המדעי:
**"Overcoming catastrophic forgetting in neural networks"** מאת Kirkpatrick et al. (DeepMind, 2016).

## סקירת הפרויקט
מטרת הפרויקט היא לשחזר את התוצאות המדעיות המציגות את אלגוריתם ה-**Elastic Weight Consolidation (EWC)**. אלגוריתם זה מאפשר לרשתות נוירונים ללמוד משימות ברצף מבלי לשכוח מידע שנלמד בעבר - תופעה המכונה "שכחה קטסטרופלית" (Catastrophic Forgetting).

## מתודולוגיה
בפרויקט זה אנו מתמקדים בניסוי ה-**Permuted MNIST**:
1. **משימה א':** אימון רשת נוירונים (MLP) על בסיס הנתונים MNIST הסטנדרטי.
2. **משימה ב':** אימון אותה הרשת על גרסה של MNIST שבה הפיקסלים עורבבו בצורה אקראית (אך עקבית).
3. **השוואה:** אנו משווים בין גישת אימון סטנדרטית (SGD) לבין אלגוריתם ה-**EWC** כדי לבחון כיצד ה-EWC משמר את הדיוק של משימה א' תוך כדי לימוד משימה ב'.

## תכונות מרכזיות
* מימוש **Fisher Information Matrix** להערכת חשיבות המשקלים ברשת.
* ויזואליזציה השוואתית של דעיכת הדיוק לעומת שימורו (Consolidation).
* מימוש ב-Python תוך שימוש בספריות למידה עמוקה מודרניות.

## מבנה המאגר
* `main.py`: הסקריפט המרכזי להרצת הניסויים.
* `EWC_Implementation.ipynb`: מחברת Jupyter עם הסברים מפורטים על הלוגיקה והקוד.
* `takeaways.pdf`: דוח רפלקטיבי על הממצאים ותהליך העבודה.
* `AI_Work_Log.md`: תיעוד האינטראקציה עם כלי AI (כמו Gemini) לאורך הפרויקט.

## הוראות הרצה
1. שיבוט המאגר (Clone).
2. התקנת ספריות נדרשות: `pip install -r requirements.txt`.
3. הרצת הניתוח: `python main.py`.

## מקורות
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.
