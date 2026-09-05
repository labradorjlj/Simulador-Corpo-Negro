import json
from flask import Flask, render_template_string, request, jsonify
import numpy as np
import plotly.graph_objects as go
import plotly.utils

app = Flask(__name__)

# Constantes Físicas
h = 6.62607015e-34  # Constante de Planck (J·s)
c = 2.99792458e8    # Velocidade da luz (m/s)
k = 1.380649e-23    # Constante de Boltzmann (J/K)
b = 2.8977719e-3    # Constante de Wien (m·K)

def planck_law(wavelength_nm, T):
    """Calcula a radiança espectral u_lambda(T) em kW / (sr * m^2 * nm)."""
    lam = wavelength_nm * 1e-9  # nm -> m
    numerator = 2 * np.pi * h * (c**2)
    denominator = (lam**5) * (np.exp((h * c) / (lam * k * T)) - 1)
    return (numerator / denominator) * 1e-9 / 1e3

def rayleigh_jeans_law(wavelength_nm, T):
    """Aproximação clássica de Rayleigh-Jeans."""
    lam = wavelength_nm * 1e-9
    return (2 * np.pi * c * k * T / (lam**4)) * 1e-9 / 1e3

def wien_approximation(wavelength_nm, T):
    """Aproximação de Wien."""
    lam = wavelength_nm * 1e-9
    numerator = 2 * np.pi * h * (c**2)
    return (numerator / (lam**5)) * np.exp(-(h * c) / (lam * k * T)) * 1e-9 / 1e3

@app.route('/')
def index():
    html_template = '''
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Simulador de Radiação de Corpo Negro (Lei de Planck)</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                padding: 20px;
                font-family: Arial, Helvetica, sans-serif;
                background: url('https://images.unsplash.com/photo-1580582932707-520aed937b7b?auto=format&fit=crop&w=1920&q=80') center/cover no-repeat fixed #333;
                color: #333;
            }
            .header-title {
                text-align: center;
                color: white;
                font-size: 26px;
                font-weight: bold;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
                margin-bottom: 20px;
            }
            .main-layout {
                display: flex;
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
                align-items: stretch;
            }
            .chart-panel {
                flex: 3;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                border: 1px solid #ccc;
            }
            .control-panel {
                flex: 1.2;
                background: rgba(245, 245, 245, 0.95);
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                border: 1px solid #ccc;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            .panel-section {
                background: white;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 12px;
            }
            .panel-title {
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 10px;
                color: #111;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }
            .slider-container {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            input[type=range] { flex: 1; }
            input[type=number] { width: 75px; padding: 4px; font-weight: bold; }
            .equation-box {
                text-align: center;
                font-size: 16px;
                background: #f9f9f9;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 8px;
                border: 1px dashed #bbb;
            }
            .constants-list {
                font-size: 12px;
                color: #444;
                line-height: 1.6;
            }
            .checkbox-group label {
                display: block;
                font-size: 13px;
                margin-bottom: 6px;
                cursor: pointer;
            }
            @media (max-width: 900px) {
                .main-layout { flex-direction: column; }
            }
        </style>
    </head>
    <body>

        <div class="header-title">
            Simulador de Radiação de Corpo Negro (Lei de Planck)
        </div>

        <div class="main-layout">
            <!-- PAINEL DO GRÁFICO (ESQUERDA) -->
            <div class="chart-panel">
                <div id="plot" style="width:100%; height:580px;"></div>
            </div>

            <!-- PAINEL DE CONTROLE (DIREITA) -->
            <div class="control-panel">
                <div class="panel-title" style="text-align: center; font-size: 18px;">
                    Painel de Controle e Parâmetros
                </div>

                <div class="panel-section">
                    <div class="panel-title">Temperatura (T)</div>
                    <div class="slider-container">
                        <input type="range" id="tempSlider" min="5" max="10000" step="10" value="6000" oninput="syncTemp('slider')">
                        <input type="number" id="tempNum" min="1" max="20000" value="6000" oninput="syncTemp('num')"> K
                    </div>
                    <p style="font-size: 11px; color: #666; margin-top: 8px; margin-bottom: 0;">
                        Ajuste a temperatura em Kelvin (K). T afeta a intensidade e o pico da radiação.
                    </p>
                </div>

                <div class="panel-section">
                    <div class="panel-title">Equação Utilizada (Lei de Planck)</div>
                    <div class="equation-box">
                        <b><i>u(λ, T) = </i></b> 
                        <span style="font-size: 17px;">(2π·h·c²) / [ λ⁵ · (e<sup>hc / λk<sub>B</sub>T</sup> - 1) ]</span>
                    </div>
                    <div class="constants-list">
                        <b>h</b> = 6,626 × 10<sup>-34</sup> J·s<br>
                        <b>c</b> = 2,998 × 10<sup>8</sup> m/s<br>
                        <b>k<sub>B</sub></b> = 1,380 × 10<sup>-23</sup> J/K
                    </div>
                </div>

                <div class="panel-section">
                    <div class="panel-title">Análise de Resultados</div>
                    <div style="font-size: 13px; line-height: 1.6;">
                        • Pico de emissão (λ<sub>max</sub>): <b><span id="resPeak">-</span></b><br>
                        • Radiança máxima: <b><span id="resInt">-</span></b><br>
                        • Cor aproximada do corpo: <b><span id="resColor">-</span></b>
                    </div>
                </div>

                <div class="panel-section">
                    <div class="panel-title">Comparação</div>
                    <div class="checkbox-group">
                        <label>
                            <input type="checkbox" id="chkRJ" onchange="updatePlot()"> 
                            Mostrar Lei de Rayleigh-Jeans (limite clássico)
                        </label>
                        <label>
                            <input type="checkbox" id="chkWien" onchange="updatePlot()"> 
                            Mostrar Lei de Wien (aproximação)
                        </label>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function syncTemp(source) {
                const slider = document.getElementById('tempSlider');
                const num = document.getElementById('tempNum');
                if (source === 'slider') {
                    num.value = slider.value;
                } else {
                    slider.value = num.value;
                }
                updatePlot();
            }

            function getColorName(T) {
                if (T < 1000) return "Infravermelho (Invisível / Calor)";
                if (T < 2500) return "Vermelho Escuro / Laranja";
                if (T < 4000) return "Amarelo (Filamento de Lâmpada)";
                if (T < 6500) return "Branco-amarelado (Sol)";
                if (T < 9000) return "Branco Puro";
                return "Azul-azulado (Estrela Quente)";
            }

            function updatePlot() {
                const temp = document.getElementById('tempNum').value;
                const showRJ = document.getElementById('chkRJ').checked;
                const showWien = document.getElementById('chkWien').checked;

                if (!temp || temp <= 0) return;

                fetch(`/get_plot?temp=${temp}&rj=${showRJ}&wien=${showWien}`)
                    .then(r => r.json())
                    .then(data => {
                        Plotly.newPlot('plot', data.data, data.layout, {responsive: true});
                        document.getElementById('resPeak').innerText = data.lambda_max_formatted;
                        document.getElementById('resInt').innerText = data.max_intensity_formatted;
                        document.getElementById('resColor').innerText = getColorName(parseFloat(temp));
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
    T = float(request.args.get('temp', 6000))
    show_rj = request.args.get('rj', 'false') == 'true'
    show_wien = request.args.get('wien', 'false') == 'true'

    # Lei de Wien
    lambda_max_nm = (b / T) * 1e9

    # Intervalo para o gráfico
    max_wave = max(2500.0, lambda_max_nm * 3.5)
    wavelengths = np.linspace(10, max_wave, 700)

    u_planck = planck_law(wavelengths, T)
    max_u = np.max(u_planck)

    fig = go.Figure()

    # Faixas do Espectro Visível no Fundo (380nm - 750nm)
    colors_visible = [
        (380, 450, 'rgba(148, 0, 211, 0.4)'),   # Violeta
        (450, 495, 'rgba(0, 0, 255, 0.4)'),     # Azul
        (495, 570, 'rgba(0, 255, 0, 0.4)'),     # Verde
        (570, 590, 'rgba(255, 255, 0, 0.4)'),   # Amarelo
        (590, 620, 'rgba(255, 127, 0, 0.4)'),   # Laranja
        (620, 750, 'rgba(255, 0, 0, 0.4)')      # Vermelho
    ]
    
    for x0, x1, col in colors_visible:
        if x0 < max_wave:
            fig.add_vrect(x0=x0, x1=min(x1, max_wave), fillcolor=col, layer="below", line_width=0)

    # Curva Principal (Planck)
    fig.add_trace(go.Scatter(
        x=wavelengths, y=u_planck, mode='lines',
        name=f'{int(T)} K (Corpo Negro)',
        line=dict(color='#d62728', width=3),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.15)'
    ))

    # Anotação com Seta apontando para o Pico
    fig.add_annotation(
        x=lambda_max_nm, y=max_u,
        text=f"Pico de emissão (λ_max = {lambda_max_nm:.1f} nm)",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black",
        ax=60, ay=-40,
        font=dict(size=12, color="black"),
        bgcolor="white", bordercolor="black", borderwidth=1
    )

    # Comparativo: Rayleigh-Jeans
    if show_rj:
        u_rj = rayleigh_jeans_law(wavelengths, T)
        fig.add_trace(go.Scatter(
            x=wavelengths, y=u_rj, mode='lines',
            name='Lei de Rayleigh-Jeans (Clássica)',
            line=dict(color='blue', width=2, dash='dash')
        ))

    # Comparativo: Wien
    if show_wien:
        u_wien = wien_approximation(wavelengths, T)
        fig.add_trace(go.Scatter(
            x=wavelengths, y=u_wien, mode='lines',
            name='Lei de Wien (Aproximação)',
            line=dict(color='green', width=2, dash='dot')
        ))

    fig.update_layout(
        title=dict(text="<b>Gráfico de Radiança Espectral (u<sub>λ</sub>) vs. Comprimento de Onda (λ)</b>", font=dict(size=16)),
        xaxis=dict(title="<b>Comprimento de Onda λ (nm)</b>", range=[0, max_wave], gridcolor="#e5e5e5"),
        yaxis=dict(title="<b>Radiança Espectral u<sub>λ</sub> (kW / (sr · m² · nm))</b>", range=[0, max_u * 1.15], gridcolor="#e5e5e5"),
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(x=0.55, y=0.95, bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1)
    )

    return jsonify({
        'data': json.loads(json.dumps(fig.data, cls=plotly.utils.PlotlyJSONEncoder)),
        'layout': json.loads(json.dumps(fig.layout, cls=plotly.utils.PlotlyJSONEncoder)),
        'lambda_max_formatted': f"~{lambda_max_nm:.0f} nm",
        'max_intensity_formatted': f"~{max_u:.1f} kW / (sr · m² · nm)"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
