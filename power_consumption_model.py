import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Función de Carga y preprocesamiento
def load_and_preprocess():
    df = pd.read_csv('powerconsumption.csv')
    
    # Convertir 'Datetime' a features temporales
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month
    df.drop('Datetime', axis=1, inplace=True)
    
    # Separar features y targets
    X = df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1)
    y_zones = {
        'Zone1': df['PowerConsumption_Zone1'],
        'Zone2': df['PowerConsumption_Zone2'],
        'Zone3': df['PowerConsumption_Zone3']
    }
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_zones, scaler, X.columns

# Entrenamiento de modelos
def train_models(X, y_zones):
    best_models = {}
    metrics_report = {}
    
    for zone, y in y_zones.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% Entrenamiento y 20% prueba 
        
        # Seleccion de modelos y definición de hiperparametro de GridSearch
        models = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}  # Multiplicacion Coef k * Feature k + ... + intercept
            },
            'Ridge Regression': {
                'model': Ridge(),
                'params': {'alpha': [0.1, 1, 10]} # Penalizacion de L2 y alfa 10%
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'params': {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
            }
        }
        
        best_models[zone] = {}
        metrics_report[zone] = {}
        
        for name, config in models.items():
            model = config['model']
            params = config['params']
            
            # Aplicar GridSearchCV si hay parámetros para optimizar
            if params:
                grid = GridSearchCV(model, params, cv=5, scoring='r2')
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                print(f"{zone} - {name}: Mejores parámetros = {grid.best_params_}")
            else:
                best_model = model.fit(X_train, y_train)
            
            # Validación cruzada con cv = 5
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
            print(f"{zone} - {name}: R2 Promedio (CV): {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
            
            # Evaluación
            y_pred = best_model.predict(X_test)
            metrics = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred)
            }
            
            best_models[zone][name] = best_model
            metrics_report[zone][name] = metrics
    
    return best_models, metrics_report

# Reporte de métricas
def save_metrics_report(metrics_report, filename='reporte_metricas.txt'):
    with open(filename, 'w') as f:
        f.write("REPORTE DE MÉTRICAS POR ZONA Y MODELO\n")
        f.write("=" * 50 + "\n")
        for zone, models in metrics_report.items():
            f.write(f"\nZONA: {zone}\n")
            f.write("-" * 50 + "\n")
            for name, metrics in models.items():
                f.write(f"\nModelo: {name}\n")
                f.write(f"  MAE: {metrics['MAE']:.2f}\n")
                f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
                f.write(f"  R2: {metrics['R2']:.4f}\n")

    # Reglas del árbol de decisión en Zona 1
    from sklearn.tree import export_text
    model = best_models['Zone1']['Decision Tree']  # o Zone2, Zone3
    rules = export_text(model, feature_names=list(X_columns))
    print(rules)


# Interfaz Gráfica
class PowerConsumptionApp:
    def __init__(self, root, X_columns, scaler, best_models):
        self.root = root
        self.root.title("Predicción de Consumo de Energía")
        self.scaler = scaler
        self.best_models = best_models
        self.X_columns = X_columns
        
        self.zone_var = tk.StringVar(value='Zone1')
        self.model_var = tk.StringVar(value='Linear Regression')
        
        self.create_widgets()
    
    def create_widgets(self):
        # Frame para inputs
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=10)
        
        # Entradas para features
        self.entries = []
        for i, feature in enumerate(self.X_columns):
            ttk.Label(input_frame, text=feature).grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(input_frame)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries.append(entry)
        
        # Dropdowns para zona y modelo
        ttk.Label(input_frame, text="Zona:").grid(row=len(self.X_columns), column=0)
        zone_dropdown = ttk.Combobox(input_frame, textvariable=self.zone_var, values=list(self.best_models.keys()))
        zone_dropdown.grid(row=len(self.X_columns), column=1)
        
        ttk.Label(input_frame, text="Modelo:").grid(row=len(self.X_columns)+1, column=0)
        model_dropdown = ttk.Combobox(input_frame, textvariable=self.model_var, values=['Linear Regression', 'Ridge Regression', 'Decision Tree'])
        model_dropdown.grid(row=len(self.X_columns)+1, column=1)
        
        # Botones
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Predecir Zona", command=self.predict_zone).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Predecir Total", command=self.predict_total).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Mostrar Gráfico", command=self.show_plot).pack(side=tk.LEFT, padx=5)
        
        # Resultado
        self.result_label = ttk.Label(self.root, text="")
        self.result_label.pack(pady=10)
        
        # Frame para el gráfico
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
    
    def predict_zone(self):
        try:
            input_data = [float(entry.get()) for entry in self.entries]
            input_scaled = self.scaler.transform([input_data])
            
            selected_zone = self.zone_var.get()
            selected_model = self.model_var.get()
            
            model = self.best_models[selected_zone][selected_model]
            prediction = model.predict(input_scaled)
            
            self.result_label.config(text=f"Consumo {selected_zone}: {prediction[0]:.2f} kW")
        except ValueError:
            self.result_label.config(text="Error: Ingresa valores numéricos")
    
    def predict_total(self):
        try:
            input_data = [float(entry.get()) for entry in self.entries]
            input_scaled = self.scaler.transform([input_data])
            total = 0
            
            for zone in self.best_models.keys():
                model = self.best_models[zone][self.model_var.get()]
                total += model.predict(input_scaled)[0]
            
            self.result_label.config(text=f"Consumo Total: {total:.2f} kW")
        except ValueError:
            self.result_label.config(text="Error: Ingresa valores numéricos")
    
    def show_plot(self):
        try:
            input_data = [float(entry.get()) for entry in self.entries]
            input_scaled = self.scaler.transform([input_data])
            
            zones = list(self.best_models.keys())
            predictions = []
            for zone in zones:
                model = self.best_models[zone][self.model_var.get()]
                predictions.append(model.predict(input_scaled)[0])
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(zones, predictions, color=['skyblue', 'lightgreen', 'salmon'])
            ax.set_title("Consumo Predicho por Zona")
            ax.set_ylabel("kW")
            
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except ValueError:
            self.result_label.config(text="Error: Ingresa valores numéricos")

# Ejecución
if __name__ == "__main__":
    X_scaled, y_zones, scaler, X_columns = load_and_preprocess()
    best_models, metrics_report = train_models(X_scaled, y_zones)
    save_metrics_report(metrics_report)
    
    root = tk.Tk()
    app = PowerConsumptionApp(root, X_columns, scaler, best_models)
    root.mainloop()