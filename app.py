import json
from flask import Flask, render_template_string, request, jsonify
import numpy as np
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)

# Constantes Físicas (SI)
h = 6.62607015e-34  # Constante de Planck (J·s)
c = 2.99792458e8    # Velocidade da luz (m/s)
k = 1.380649e-23    # Constante de Boltzmann (J/K)
b = 2.8977719e-3    # Constante de Wien (m·K)

def planck_law(wavelength_nm, T):
    """Calcula a radiança espectral pela Lei de Planck."""
    lam = wavelength_nm * 1e-9  # Converte nanômetros para metros
    numerator = 2 * h * c**2
    denominator = (lam**5) * (np.exp((h * c) / (lam * k * T)) - 1)
    # Retorna o valor na escala kW / (sr * m^2 * nm)
    return (numerator / denominator) * 1e-9 / 1e3

def get_spectrum_region(lambda_nm):
    """Determina a faixa do espectro eletromagnético segundo o comprimento de onda pico."""
    if lambda_nm < 380:
        return "Ultraviolete / Raios-X / Gama"
    elif 380 <= lambda_nm <= 750:
        return "Luz Visível"
    elif 750 < lambda_nm <= 1e6:
        return "Infravermelho"
    else:
        return "Micro-ondas / Ondas de Rádio"

@app.route('/')
def index():
    html_template = '''
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Simulador de Radiação de Corpo Negro</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 30px; 
                background-color: #f0f2f5; 
            }
            .container { 
                max-width: 950px; 
                margin: 0 auto; 
                background: white; 
                padding: 25px; 
                border-radius: 12px; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
            }
            .controls { 
                margin-bottom: 20px; 
                display: flex; 
                align-items: center; 
                gap: 12px; 
                flex-wrap: wrap; 
            }
            input[type=number] { 
                padding: 8px 12px; 
                font-size: 1em; 
                border: 1px solid #ccc; 
                border-radius: 6px; 
                width: 140px; 
            }
            button { 
                padding: 8px 18px; 
                font-size: 1em; 
                background-color: #007bff; 
                color: white; 
                border: none; 
                border-radius: 6px; 
                cursor: pointer; 
                transition: background-color 0.2s;
            }
            button:hover { 
                background-color: #0056b3; 
            }
            .info-box { 
                margin-top: 20px; 
                padding: 15px; 
                background-color: #f8f9fa; 
                border-left: 4px solid #007bff;
                border-radius: 4px; 
                font-size: 1.05em; 
                line-height: 1.6; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Simulador de Radiação de Corpo Negro</h2>
            <div class="controls">
                <label for="tempInput"><strong>Temperatura (K):</strong></label>
                <input type="number" id="tempInput" value="5" min="0.001" step="any" placeholder="Ex: 5, 5800">
                <button onclick="updatePlot()">Gerar Gráfico</button>
            </div>

            <div id="plot"></div>

            <div class="info-box" id="infoBox">
                <strong>Resultados Calculados:</strong><br>
                • <strong>Pico de Emissão ($\lambda_{max}$):</strong> <span id="wienPeak">-</span><br>
                • <strong>Região do Espectro:</strong> <span id="spectrumRegion">-</span>
            </div>
        </div>

        <script>
            function updatePlot() {
                const temp = document.getElementById('tempInput').value;
                if (!temp || temp <= 0) {
                    alert('Por favor, insira uma temperatura válida maior que zero em Kelvin.');
                    return;
                }

                fetch('/get_plot?temp=' + temp)
                    .then(response => response.json())
                    .then(data => {
                        Plotly.newPlot('plot', data.data, data.layout);
                        document.getElementById('wienPeak').innerText = data.lambda_max_formatted;
                        document.getElementById('spectrumRegion').innerText = data.region;
                    });
            }
            window.onload = updatePlot;
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/get_plot')
def get_plot():
    # Obtém a temperatura do parâmetro de URL (padrão é 5 K)
    T = float(request.args.get('temp', 5))
    
    # Lei de Deslocamento de Wien: lambda_max = b / T
    lambda_max_m = b / T
    lambda_max_nm = lambda_max_m * 1e9

    # Ajuste automático do intervalo do eixo X em função do pico de emissão
    min_wave = max(1e-3, lambda_max_nm * 0.05)
    max_wave = lambda_max_nm * 3.5
    
    wavelengths = np.linspace(min_wave, max_wave, 600)
    intensity = planck_law(wavelengths, T)

    fig = go.Figure()
    
    # Plotagem da Curva de Planck
    fig.add_trace(go.Scatter(
        x=wavelengths, 
        y=intensity, 
        mode='lines', 
        name=f'{T} K', 
        line=dict(color='#d62728', width=3)
    ))

    # Marcação do Ponto de Pico (Lei de Wien)
    pico_intensidade = planck_law(np.array([lambda_max_nm]), T)[0]
    fig.add_trace(go.Scatter(
        x=[lambda_max_nm], 
        y=[pico_intensidade],
        mode='markers+text',
        name='Pico ($\lambda_{max}$)',
        text=[f'Pico: {lambda_max_nm:.2e} nm'],
        textposition="top center",
        marker=dict(color='black', size=10)
    ))

    fig.update_layout(
        title=f'Espectro de Radiação de Corpo Negro (T = {T} K)',
        xaxis_title='Comprimento de Onda λ (nm)',
        yaxis_title='Radiança Espectral (kW / (sr·m²·nm))',
        template='plotly_white'
    )

    # Formatação amigável do comprimento de onda
    if lambda_max_nm >= 1e6:
        lambda_str = f"{lambda_max_m:.4e} m ({lambda_max_nm:.2e} nm)"
    else:
        lambda_str = f"{lambda_max_nm:.2f} nm ({lambda_max_m:.4e} m)"

    region = get_spectrum_region(lambda_max_nm)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({
        'data': json.loads(graphJSON)['data'],
        'layout': json.loads(graphJSON)['layout'],
        'lambda_max_formatted': lambda_str,
        'region': region
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
