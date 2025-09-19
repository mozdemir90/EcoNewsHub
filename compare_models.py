import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support, cohen_kappa_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
import subprocess
import sys

class ModelComparator:
    def __init__(self):
        self.results = {}
        self.test_data = None
        self.pred_store = {}
        
    def load_test_data(self):
        """Test verilerini yükle"""
        print("📥 Test verileri yükleniyor...")
        import glob
        # Prefer analiz_sonuclari3 if available, else fallback to 2
        v = 3 if glob.glob('data/analiz_sonuclari3_tahminli_*.xlsx') else 2
        files = {
            'TF-IDF': f'data/analiz_sonuclari{v}_tahminli_TF-IDF.xlsx',
            'Word2Vec': f'data/analiz_sonuclari{v}_tahminli_w2v.xlsx',
            'GloVe': f'data/analiz_sonuclari{v}_tahminli_glove.xlsx',
            'Deep Learning': f'data/analiz_sonuclari{v}_tahminli_DL.xlsx',
        }
        
        self.test_data = {}
        for name, file_path in files.items():
            if os.path.exists(file_path):
                self.test_data[name] = pd.read_excel(file_path)
                print(f"✅ {name}: {len(self.test_data[name])} satır")
            else:
                print(f"❌ {name}: Dosya bulunamadı - {file_path}")
        
        # Orijinal test verisi
        if os.path.exists(f'data/analiz_sonuclari{v}.xlsx'):
            self.original_test = pd.read_excel(f'data/analiz_sonuclari{v}.xlsx')
            print(f"✅ Orijinal test: {len(self.original_test)} satır")
        else:
            print("❌ Orijinal test verisi bulunamadı")
            return False
            
        return True
    
    def calculate_metrics(self, y_true, y_pred):
        """Regresyon + sınıflandırma metriklerini hesapla (1..5 sıralı sınıflar)."""
        # Regresyon benzeri
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Sınıflandırma için tamsayı etiketler
        y_true_int = np.clip(np.rint(y_true).astype(int), 1, 5)
        y_pred_int = np.clip(np.rint(y_pred).astype(int), 1, 5)

        acc = accuracy_score(y_true_int, y_pred_int)
        pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(y_true_int, y_pred_int, average='macro', zero_division=0)
        pr_weighted, rc_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true_int, y_pred_int, average='weighted', zero_division=0)
        try:
            kappa_qw = cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic')
        except Exception:
            kappa_qw = cohen_kappa_score(y_true_int, y_pred_int)

        return {
            'MSE': mse,
            'MAE': mae,
            'R²': r2,
            'RMSE': np.sqrt(mse),
            'ACC': acc,
            'F1_macro': f1_macro,
            'F1_weighted': f1_weighted,
            'P_macro': pr_macro,
            'R_macro': rc_macro,
            'Kappa_qw': kappa_qw
        }
    
    def evaluate_models(self):
        """Tüm modelleri değerlendir. Her yöntem altındaki TÜM varyantları (rf/svm/nb/ada/ann, w2v/glove varyant isimleri, dl cnn/lstm) ayrı ayrı raporlar."""
        print("\n📊 Model değerlendirmesi başlıyor...")
        
        varliklar = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
        
        for model_name, test_df in self.test_data.items():
            print(f"\n🎯 {model_name} değerlendiriliyor...")
            
            # Her varyantı ayrı bir 'Model' olarak raporlamak için dış anahtarı genişletiyoruz
            # Örn: "TF-IDF:rf", "TF-IDF:svm", "GloVe:rf", "Deep Learning:cnn" vb.
            collected_variant_names = set()
            
            for varlik in varliklar:
                y_true = self.original_test[varlik].values
                # Varyant sütunlarını seç
                if model_name == 'Word2Vec':
                    pred_columns = [col for col in test_df.columns if col.startswith(varlik + '_') and 'w2v' in col]
                elif model_name == 'GloVe':
                    pred_columns = [col for col in test_df.columns if col.startswith(varlik + '_') and 'glove' in col]
                elif model_name == 'TF-IDF':
                    pred_columns = [col for col in test_df.columns if col.startswith(varlik + '_') and ('rf' in col or 'svm' in col or 'nb' in col or 'ada' in col or 'ann' in col) and 'w2v' not in col and 'glove' not in col]
                elif model_name == 'Deep Learning':
                    pred_columns = [col for col in test_df.columns if col.startswith(varlik + '_') and ('cnn' in col or 'lstm' in col)]
                else:
                    pred_columns = [col for col in test_df.columns if col.startswith(varlik + '_') and col != varlik]
                
                if not pred_columns:
                    print(f"  {varlik}: Tahmin sütunu bulunamadı")
                    continue
                
                for col in pred_columns:
                    y_pred = test_df[col].values
                    metrics = self.calculate_metrics(y_true, y_pred)
                    # Varyant adını sütun adından çıkar
                    variant = col.replace(varlik + '_', '')
                    # DL sütunları genellikle "cnn_dl" gibi olabilir
                    variant = variant.replace(varlik, '')
                    variant_key = f"{model_name}:{variant}"
                    collected_variant_names.add(variant_key)
                    # Sonuçları toplu yapıda sakla
                    if variant_key not in self.results:
                        self.results[variant_key] = {}
                    self.results[variant_key][varlik] = metrics
                    # Confusion matrix için gerçek ve tahminleri sakla
                    self.pred_store[(variant_key, varlik)] = (y_true, y_pred)
                    print(f"  {varlik} [{variant_key}]: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R²']:.4f}, F1(macro)={metrics['F1_macro']:.4f}, ACC={metrics['ACC']:.4f}, KappaQW={metrics['Kappa_qw']:.4f}")
    
    def create_comparison_table(self):
        """Karşılaştırma tablosu oluştur"""
        print("\n📋 Karşılaştırma tablosu oluşturuluyor...")
        
        comparison_data = []
        
        for model_name, model_results in self.results.items():
            for varlik, metrics in model_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Varlık': varlik,
                    'MSE': metrics['MSE'],
                    'MAE': metrics['MAE'],
                    'R²': metrics['R²'],
                    'RMSE': metrics['RMSE'],
                    'ACC': metrics.get('ACC'),
                    'F1_macro': metrics.get('F1_macro'),
                    'F1_weighted': metrics.get('F1_weighted'),
                    'Kappa_qw': metrics.get('Kappa_qw')
                })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Excel'e kaydet
        self.comparison_df.to_excel('data/model_comparison.xlsx', index=False)
        print("✅ Karşılaştırma tablosu kaydedildi: data/model_comparison.xlsx")
        
        return self.comparison_df
    
    def create_visualizations(self):
        """Görselleştirmeler oluştur"""
        print("\n📈 Görselleştirmeler oluşturuluyor...")
        
        # Matplotlib ayarları
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performans Karşılaştırması', fontsize=16, fontweight='bold')
        
        # Metrikler
        metrics = ['MSE', 'MAE', 'R²', 'RMSE']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Pivot table oluştur
            pivot_data = self.comparison_df.pivot(index='Varlık', columns='Model', values=metric)
            
            # Heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlBu_r', ax=ax)
            ax.set_title(f'{metric} Karşılaştırması')
            ax.set_xlabel('Model')
            ax.set_ylabel('Varlık')
        
        plt.tight_layout()
        plt.savefig('data/model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        print("✅ Heatmap kaydedildi: data/model_comparison_heatmap.png")
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        
        # R² skorları için bar plot
        r2_data = self.comparison_df[self.comparison_df['R²'].notna()]
        
        if not r2_data.empty:
            sns.barplot(data=r2_data, x='Varlık', y='R²', hue='Model')
            plt.title('R² Skorları Karşılaştırması')
            plt.xticks(rotation=45)
            plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('data/r2_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ R² karşılaştırması kaydedildi: data/r2_comparison.png")

    def create_confusion_matrices(self):
        """Her model varyantı ve varlık için confusion matrix üretir ve kaydeder."""
        print("\n🧩 Confusion matrix'ler oluşturuluyor...")
        import os
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        os.makedirs('data/confusion_matrices', exist_ok=True)
        classes = [1,2,3,4,5]
        for (variant_key, varlik), (y_true, y_pred) in self.pred_store.items():
            try:
                cm = confusion_matrix(y_true, y_pred, labels=classes)
                plt.figure(figsize=(4.5,3.8))
                sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues',
                            xticklabels=classes, yticklabels=classes)
                plt.title(f'{variant_key} | {varlik}')
                plt.xlabel('Tahmin')
                plt.ylabel('Gerçek')
                safe_variant = variant_key.replace(' ', '_').replace(':','_')
                safe_varlik = varlik
                out_path = f'data/confusion_matrices/{safe_variant}__{safe_varlik}.png'
                plt.tight_layout()
                plt.savefig(out_path, dpi=200)
                plt.close()
            except Exception as exc:
                print(f"Confusion matrix hatası {variant_key}-{varlik}: {exc}")
    
    def generate_report(self):
        """Rapor oluştur"""
        print("\n📄 Rapor oluşturuluyor...")
        
        report = []
        report.append("# Model Performans Karşılaştırma Raporu")
        report.append("="*50)
        report.append("")
        
        # En iyi model her varlık için
        for varlik in ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']:
            varlik_data = self.comparison_df[self.comparison_df['Varlık'] == varlik]
            
            if not varlik_data.empty:
                best_model = varlik_data.loc[varlik_data['R²'].idxmax()]
                report.append(f"## {varlik.replace('_skor', '').title()}")
                report.append(f"**En İyi Model:** {best_model['Model']}")
                report.append(f"**R² Skoru:** {best_model['R²']:.4f}")
                report.append(f"**MAE:** {best_model['MAE']:.4f}")
                report.append(f"**RMSE:** {best_model['RMSE']:.4f}")
                report.append("")
        
        # Genel özet
        report.append("## Genel Özet")
        overall_best = self.comparison_df.loc[self.comparison_df['R²'].idxmax()]
        report.append(f"**En İyi Genel Performans:** {overall_best['Model']} - {overall_best['Varlık']}")
        report.append(f"**R² Skoru:** {overall_best['R²']:.4f}")
        report.append("")
        
        # Model bazında ortalama performans
        report.append("## Model Bazında Ortalama Performans")
        model_avg = self.comparison_df.groupby('Model')['R²'].mean().sort_values(ascending=False)
        for model, avg_r2 in model_avg.items():
            report.append(f"- **{model}:** R² = {avg_r2:.4f}")
        
        # Raporu kaydet
        with open('data/model_comparison_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("✅ Rapor kaydedildi: data/model_comparison_report.md")
        
    def append_cv_summaries(self):
        """TF-IDF ve W2V/GloVe 5-fold CV çıktılarını rapora ekle"""
        print("\n🔁 5-fold CV özetleri çalıştırılıyor...")
        outputs = []
        cmds = [
            [sys.executable, 'trainTF-IDF.py'],
            [sys.executable, 'trainWord2Vec_GloVe.py'],
        ]
        for cmd in cmds:
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
                text = res.stdout
                filtered = '\n'.join([line for line in text.splitlines() if '[CV5]' in line or '[VAL]' in line])
                outputs.append((cmd[1], filtered))
            except Exception as exc:
                outputs.append((cmd[1], f"hata: {exc}"))
        with open('data/model_comparison_report.md', 'a', encoding='utf-8') as f:
            f.write("\n\n## 5-fold CV Özetleri (Eğitim içi)\n")
            for name, out in outputs:
                f.write(f"\n### {name}\n")
                f.write("""```
""")
                f.write(out.strip() + "\n")
                f.write("""```
""")
    
    def run_comparison(self):
        """Tüm karşılaştırma sürecini çalıştır"""
        print("🎯 Model Karşılaştırma Süreci Başlıyor...")
        print("="*60)
        
        # 1. Test verilerini yükle
        if not self.load_test_data():
            print("❌ Test verileri yüklenemedi!")
            return
        
        # 2. Modelleri değerlendir
        self.evaluate_models()
        
        # 3. Karşılaştırma tablosu oluştur
        self.create_comparison_table()
        
        # 4. Görselleştirmeler oluştur
        self.create_visualizations()
        # 4b. Confusion matrix görselleri
        self.create_confusion_matrices()
        
        # 5. Rapor oluştur
        self.generate_report()
        # 6. CV özetlerini ekle
        self.append_cv_summaries()
        
        print("\n🎉 Model karşılaştırması tamamlandı!")
        print("📁 Sonuçlar:")
        print("- data/model_comparison.xlsx")
        print("- data/model_comparison_heatmap.png")
        print("- data/r2_comparison.png")
        print("- data/model_comparison_report.md")

if __name__ == "__main__":
    # Karşılaştırmayı başlat
    comparator = ModelComparator()
    comparator.run_comparison() 