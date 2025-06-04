import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import cycle

# Tekrarlanabilirlik için rastgele tohum belirle
np.random.seed(42)

# Çıktıların kaydedileceği klasörü oluştur
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Veri setini yükle
data = pd.read_csv('ObesityDataSet.csv')

# Veri Ön İşleme
# Kategorik değişkenleri one-hot encoding ile sayısallaştır
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Hedef değişkeni (NObeyesdad) kodla
le = LabelEncoder()
y = le.fit_transform(data_encoded['NObeyesdad'])
X = data_encoded.drop('NObeyesdad', axis=1)

# Sayısal öznitelikleri standartlaştır
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Veriyi eğitim ve test setlerine ayır (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Random Forest modelini oluştur ve eğit
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = rf_model.predict(X_test)

# Değerlendirme Metriklerini Hesapla
# Doğruluk (accuracy), duyarlılık (recall), hassasiyet (precision) ve F1-skoru
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Özgüllük (specificity) hesaplama
cm = confusion_matrix(y_test, y_pred)
specificity_scores = []
for i in range(len(cm)):
    tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]  # Gerçek negatif
    fp = np.sum(cm[:, i]) - cm[i, i]  # Yanlış pozitif
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_scores.append(specificity)
specificity = np.mean(specificity_scores)

# Metrikleri ekrana yazdır
print("Değerlendirme Metrikleri:")
print(f"Doğruluk: {accuracy:.4f}")
print(f"Duyarlılık (Recall): {sensitivity:.4f}")
print(f"Özgüllük: {specificity:.4f}")
print(f"F1-Skoru: {f1:.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 5 katlı çapraz doğrulama ile modelin genelleme performansını değerlendir
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"5 Katlı Çapraz Doğrulama Doğruluğu: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Palechor ve Manotas (2019) ile karşılaştırma
print("\nPalechor ve Manotas (2019) ile Karşılaştırma:")
print("Palechor ve diğerleri, SVM/Lojistik Regresyon ile ~%83 doğruluk bildirdi.")
print(f"Bizim Random Forest modelimiz {accuracy:.4f} doğruluk elde etti.")

# Görselleştirmeler
# 1. Karmaşıklık Matrisi
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# 2. Özellik Önem Sıralaması (En önemli 10 öznitelik)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')
plt.close()

# 3. ROC Eğrileri (Çok sınıflı)
plt.figure(figsize=(10, 8))
y_score = rf_model.predict_proba(X_test)
n_classes = len(le.classes_)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, label=f'ROC curve of class {le.classes_[i]} (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Multi-class')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('outputs/roc_curve.png')
plt.close()

# 4. Hassasiyet-Geri Çağırma Eğrileri
plt.figure(figsize=(10, 8))
for i, color in zip(range(n_classes), colors):
    precision, recall, _ = precision_recall_curve(y_test == i, y_score[:, i])
    plt.plot(recall, precision, color=color, label=f'PR curve of class {le.classes_[i]}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Multi-class')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig('outputs/pr_curve.png')
plt.close()

# 5. Özellik Korelasyon Matrisi
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('outputs/correlation_matrix.png')
plt.close()

# 6. En Önemli 3 Özelliğin Çift Grafiği
top_features = feature_importance['Feature'].head(3).tolist()
pair_data = data_encoded[top_features + ['NObeyesdad']].copy()
pair_data['NObeyesdad'] = le.inverse_transform(y)
sns.pairplot(pair_data, hue='NObeyesdad', diag_kind='hist')
plt.suptitle('Pairplot of Top 3 Features', y=1.02)
plt.savefig('outputs/pairplot.png')
plt.close()

# 7. Öğrenme Eğrisi
train_sizes, train_scores, test_scores = learning_curve(rf_model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Cross-validation score')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('outputs/learning_curve.png')
plt.close()

# 8. Çapraz Doğrulama Skorları Kutu Grafiği
plt.figure(figsize=(8, 6))
sns.boxplot(y=cv_scores)
plt.title('5-Fold Cross-Validation Accuracy Scores')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('outputs/cv_boxplot.png')
plt.close()

# 9. Sınıf Bazlı Öznitelik Dağılımları (İlk 3 sayısal öznitelik)
for feature in numerical_cols[:3]:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=le.inverse_transform(y), y=data[feature])
    plt.title(f'Distribution of {feature} by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'outputs/feature_dist_{feature}.png')
    plt.close()

# Metrikleri ve raporları dosyaya kaydet
with open('outputs/evaluation_metrics.txt', 'w') as f:
    f.write("Değerlendirme Metrikleri:\n")
    f.write(f"Doğruluk: {accuracy:.4f}\n")
    f.write(f"Duyarlılık (Recall): {sensitivity:.4f}\n")
    f.write(f"Özgüllük: {specificity:.4f}\n")
    f.write(f"F1-Skoru: {f1:.4f}\n")
    f.write("\nSınıflandırma Raporu:\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_))
    f.write("\n5 Katlı Çapraz Doğrulama Doğruluğu: {:.4f} (±{:.4f})\n".format(cv_scores.mean(), cv_scores.std()))
    f.write("\nPalechor ve Manotas (2019) ile Karşılaştırma:\n")
    f.write("Palechor ve diğerleri, SVM/Lojistik Regresyon ile ~%83 doğruluk bildirdi.\n")
    f.write(f"Bizim Random Forest modelimiz {accuracy:.4f} doğruluk elde etti.")

# Kaydedilen çıktıları listele
print("\nKaydedilen Çıktılar:")
for file in os.listdir('outputs'):
    print(f"- {file}")