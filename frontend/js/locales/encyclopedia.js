(function () {
  const EN = {
    '1. Anatomía del Altavoz Dinámico': '1. Anatomy of a Dynamic Driver',
    'El transductor de radiación directa es el corazón de cualquier sistema. Su función es convertir la energía eléctrica en energía acústica audible.': 'The direct-radiating transducer is the heart of any loudspeaker system. It converts electrical energy into audible acoustic energy.',
    'Componentes clave': 'Key components',
    'Motor Magnético:': 'Magnetic Motor:',
    'Imán permanente + piezas polares. Crea el campo estático en el entrehierro.': 'Permanent magnet and pole pieces. They create the static magnetic field in the gap.',
    'Bobina Móvil:': 'Voice Coil:',
    'Devanado de cobre en el campo magnético. La señal alterna genera fuerza electromotriz.': 'Copper winding inside the magnetic field. The alternating signal produces a driving force.',
    'Diafragma (Cono):': 'Diaphragm (Cone):',
    'Pistón que empuja el aire. Debe ser rígido y ligero.': 'The piston that moves the air. It must be rigid and lightweight.',
    'Suspensión + Araña:': 'Surround + Spider:',
    'Resorte mecánico que centra la bobina y devuelve el cono al reposo.': 'The mechanical spring that centers the voice coil and returns the cone to rest.',
    'El conjunto móvil (cono + bobina) tiene': 'The moving assembly (cone + voice coil) has',
    'masa': 'mass', ', y la suspensión tiene': ', while the suspension has', 'elasticidad': 'compliance',
    '. Esto forma un resonador masa-resorte con una frecuencia natural de resonancia (': '. Together they form a mass-spring resonator with a natural resonant frequency (',
    'Ir a la Calculadora': 'Go to the Calculator',
    'Diseña la caja para tu altavoz con los parámetros T/S': 'Design an enclosure for your driver using its T/S parameters',

    '2. Modelo Electro-Mecánico-Acústico': '2. Electro-Mechanical-Acoustic Model',
    'Para diseñar una caja correctamente, tratamos al altavoz como un': 'To design an enclosure correctly, we treat the driver as an',
    'circuito equivalente': 'equivalent circuit',
    '. La energía pasa por tres dominios:': '. Energy flows through three domains:',
    '1. Eléctrico': '1. Electrical',
    'Voltaje del amplificador, Re (resistencia DC)': 'Amplifier voltage, Re (DC resistance)',
    '2. Mecánico': '2. Mechanical',
    'Masa (Mms), Resorte (Cms), Fricción (Rms)': 'Mass (Mms), Compliance (Cms), Damping (Rms)',
    '3. Acústico': '3. Acoustic',
    'Presión, Compliancia acústica (Cas), Masa acústica': 'Pressure, acoustic compliance (Cas), acoustic mass',
    'El factor de fuerza': 'The force factor',
    '(Teslas × metros) es el "engranaje" que convierte la corriente eléctrica en fuerza mecánica. Un Bl alto significa un motor potente — y se refleja en un Qes bajo.': '(tesla-metres) is the coupling that converts electrical current into mechanical force. A high Bl indicates a powerful motor and results in a low Qes.',
    'Calcular con estos parámetros': 'Calculate with these parameters',
    'Pon a prueba el modelo con tu altavoz real': 'Test the model with your actual driver',

    '3. Parámetros Thiele-Small': '3. Thiele-Small Parameters',
    'Sin ellos el diseño de la caja es pura adivinanza. Haz clic en cada tarjeta para explorar su función técnica.': 'Without them, enclosure design is guesswork. Select each card to explore its technical role.',
    'Resonancia Libre': 'Free-Air Resonance',
    'Frecuencia (Hz) donde la masa del cono resuena con su suspensión al aire libre. Por debajo de Fs la respuesta cae abruptamente. La caja sube esta frecuencia al añadir rigidez neumática.': 'The frequency (Hz) at which the cone mass resonates with its suspension in free air. Response drops sharply below Fs. An enclosure raises this frequency by adding pneumatic stiffness.',
    'Factor de Calidad Total': 'Total Quality Factor',
    'Describe el amortiguamiento del altavoz en Fs.': 'Describes driver damping at Fs.',
    'motor fuerte, ideal Bass-Reflex.': 'strong motor, ideal for bass reflex.',
    'motor débil, necesita la rigidez de caja sellada.': 'weaker motor, benefits from the stiffness of a sealed enclosure.',
    'Vol. Acústico Equiv.': 'Equivalent Acoustic Volume',
    'Elasticidad de la suspensión expresada en litros de aire.': 'Suspension compliance expressed as an equivalent volume of air.',
    'NO': 'NOT',
    'es el volumen de la caja. Vas alto = suspensión blanda = caja grande necesaria.': 'the enclosure volume. High Vas means a compliant suspension and usually requires a larger enclosure.',
    'Excursión Máxima': 'Maximum Excursion',
    'Desplazamiento lineal máximo (mm) en una dirección sin salir del campo magnético. Determina el SPL máximo en graves sin distorsión.': 'Maximum linear travel (mm) in one direction while the voice coil remains within the magnetic field. It determines distortion-free maximum bass SPL.',
    'Área Efectiva': 'Effective Piston Area',
    'Superficie radiante del cono (cm²). Con Xmax define el': 'Effective radiating cone area (cm²). Together with Xmax, it defines',
    'Volumen de Desplazamiento': 'volume displacement',
    ': Vd = Sd × Xmax. Mayor Sd = más aire movido a igual excursión.': ': Vd = Sd × Xmax. Greater Sd moves more air at the same excursion.',
    'Efficiency Bandwidth Product': 'Efficiency Bandwidth Product',
    'EBP = Fs/Qes. Guía rápida de tipo de recinto ideal:': 'EBP = Fs/Qes. A quick guide to the most suitable enclosure type:',
    'Bass-Reflex claro.': 'clearly favors bass reflex.',
    'ambos tipos funcionan.': 'either enclosure type can work.',
    'Caja sellada.': 'sealed enclosure.',
    'Masa Móvil': 'Moving Mass',
    'Masa total del conjunto móvil (cono + bobina + suspensión) en gramos.': 'Total mass of the moving assembly (cone, voice coil, and suspension) in grams.',
    'Crítico para cálculos precisos.': 'Critical for accurate calculations.',
    'Si se omite, el simulador lo estima desde Vas y Sd, pero con errores que pueden superar el 100%.': 'If omitted, the simulator estimates it from Vas and Sd, but the error can exceed 100%.',
    '• Mms bajo + Bl alto = alta eficiencia': '• Low Mms + high Bl = high efficiency',
    '• Mms alto = mayor inercia, menor sensibilidad': '• High Mms = greater inertia and lower sensitivity',
    'Factor de Fuerza': 'Force Factor',
    'Producto del campo magnético (B) por la longitud del hilo en el entrehierro (l), en Tesla·metro. Es el "engranaje" entre el dominio eléctrico y el mecánico:': 'Product of magnetic flux density (B) and wire length in the gap (l), measured in tesla-metres. It is the coupling between the electrical and mechanical domains:',
    '• Bl alto → Qes bajo → motor potente': '• High Bl → low Qes → powerful motor',
    '• Bl bajo → Qes alto → motor débil': '• Low Bl → high Qes → weaker motor',
    '• Afecta directamente la excursión calculada': '• Directly affects calculated excursion',
    'Introducir mis parámetros T/S en la calculadora': 'Enter my T/S parameters in the calculator',
    'Calcula Vb, Fb, F3 y planos de corte': 'Calculate Vb, Fb, F3, and cut plans',

    '4. Diseño de Cajas Cerradas': '4. Sealed Enclosure Design',
    'También llamadas': 'Also known as',
    '. El aire encerrado actúa como resorte neumático adicional.': '. The enclosed air acts as an additional pneumatic spring.',
    'Características': 'Characteristics', 'Caída:': 'Roll-off:',
    '12 dB/octava por debajo de Fc': '12 dB/octave below Fc',
    'Ventajas:': 'Advantages:',
    'Bajos rápidos y precisos. El cono no se descontrola a ultrabajas.': 'Fast, accurate bass. Cone motion remains controlled at very low frequencies.',
    'Desventajas:': 'Disadvantages:',
    'Menor extensión en graves que Reflex del mismo tamaño.': 'Less bass extension than a reflex enclosure of the same size.',
    'Altavoces ideales:': 'Suitable drivers:', 'Qts 0.4–0.7, Vas alto.': 'Qts 0.4–0.7, high Vas.',
    'La caja eleva la resonancia libre Fs al valor Fc del sistema, con Qtc que determina la forma de la curva.': 'The enclosure raises free-air resonance Fs to the system resonance Fc, while Qtc determines the response shape.',
    'Qtc = 0.707 (Butterworth) es el compromiso óptimo: máxima planura sin pico. Qtc < 0.577 tiene caída suave pero el altavoz necesita una caja muy grande.': 'Qtc = 0.707 (Butterworth) is the optimal compromise: maximally flat response without a peak. Qtc < 0.577 rolls off gently but requires a very large enclosure.',
    'Diseñar una caja cerrada': 'Design a sealed enclosure',
    'Abre la calculadora con tipo = Sellada preseleccionado': 'Open the calculator with Sealed preselected',

    '5. Cajas Bass-Reflex (Ventiladas)': '5. Bass Reflex (Vented) Enclosures',
    'Aprovechan la onda trasera del cono a través de un': 'They use the cone’s rear wave through a',
    'resonador de Helmholtz': 'Helmholtz resonator',
    'para reforzar la salida en graves.': 'to reinforce low-frequency output.',
    'Comportamiento en Fb': 'Behavior at Fb',
    'A la frecuencia de sintonía Fb, la': 'At the tuning frequency Fb, the',
    'masa de aire del tubo oscila violentamente': 'air mass in the port oscillates strongly',
    'y produce casi todo el sonido. El cono apenas se mueve — excursión mínima — lo que protege el altavoz.': 'and produces nearly all the output. Cone motion is minimal, which protects the driver.',
    'Peligro por debajo de Fb': 'Risk below Fb',
    'Sin contrapresión, el cono se mueve sin control. El altavoz puede exceder su Xmax y dañarse. Por eso se recomienda un': 'Without acoustic loading, cone motion becomes uncontrolled. The driver can exceed Xmax and be damaged. This is why a',
    'filtro subsónico': 'subsonic filter',
    'a 0.7×Fb.': 'at 0.7×Fb is recommended.',
    'Caída: 24 dB/octava por debajo de Fb': 'Roll-off: 24 dB/octave below Fb',
    'Mayor extensión y eficiencia que caja sellada del mismo tamaño': 'Greater extension and efficiency than a sealed enclosure of the same size',
    'Respuesta transitoria algo menor que sellada': 'Slightly poorer transient response than a sealed enclosure',
    'Diseñar un Bass-Reflex': 'Design a bass reflex enclosure',
    'Calcula Vb, Fb, longitud del puerto y velocidad del aire': 'Calculate Vb, Fb, port length, and port air velocity',

    '6. Diseños Avanzados': '6. Advanced Enclosure Designs',
    'Paso-Banda (4°)': 'Bandpass (4th order)', 'Línea de Transmisión': 'Transmission Line', 'Paso-Banda': 'Bandpass',
    'El transductor está completamente oculto. La cámara delantera actúa como filtro acústico pasa-bajos. Eficiencia altísima en una banda estrecha.': 'The driver is fully enclosed. The front chamber acts as an acoustic low-pass filter, providing very high efficiency over a narrow bandwidth.',
    'Todo el sonido sale por el puerto': 'All sound exits through the port',
    'Sin crossover para filtrar agudos': 'No crossover is needed to filter high frequencies',
    'Casi exclusivo para subwoofers': 'Used almost exclusively for subwoofers',
    'Respuesta transitoria pobre si mal sintonizado': 'Poor transient response when incorrectly tuned',
    'La onda trasera recorre un laberinto largo y densamente amortiguado. La longitud equivale a λ/4 de la frecuencia de refuerzo.': 'The rear wave travels through a long, heavily damped path. Its length is one quarter wavelength at the reinforcement frequency.',
    'Impedancia eléctrica muy plana': 'Very flat electrical impedance',
    'Bajos inmensamente profundos y limpios': 'Exceptionally deep, clean bass',
    'Recintos enormes y complejos de calcular': 'Very large enclosures that are complex to calculate',

    '7. Difracción de Bordes y Baffle Step': '7. Edge Diffraction and Baffle Step',
    'El baffle no es solo soporte mecánico: es un componente acústico crítico.': 'The baffle is more than mechanical support: it is a critical acoustic component.',
    'El efecto Baffle Step': 'The Baffle Step Effect', 'Agudos': 'High frequencies',
    '(longitud de onda corta): se propagan hacia el frente de forma': '(short wavelengths) radiate forward in a',
    'direccional': 'directional', '. Energía concentrada.': ' pattern, concentrating energy.', 'Graves': 'Low frequencies',
    '(longitud de onda larga): "se doblan" alrededor del recinto y se propagan': '(long wavelengths) bend around the enclosure and radiate',
    'omnidireccionalmente': 'omnidirectionally', '. Menor energía frontal.': ', reducing forward energy.',
    'Resultado: caída intrínseca de ~6 dB en graves que se debe compensar en el crossover.': 'The result is an inherent ~6 dB reduction in bass output that must be compensated in the crossover.',
    'La frecuencia del escalón es aproximadamente:': 'The step frequency is approximately:',
    'Solución práctica:': 'Practical solution:',
    'Siempre biselar o redondear fuertemente los bordes frontales del recinto. Reduce las difracciones y suaviza las irregularidades en la respuesta de medios-agudos.': 'Always chamfer or heavily round the front edges of the enclosure. This reduces diffraction and smooths irregularities in the mid/high-frequency response.',

    '8. Materiales y Proporciones': '8. Materials and Proportions', 'Materiales': 'Materials',
    'MDF 18mm:': '18mm MDF:', 'Estándar de la industria. Denso, isótropo, fácil de mecanizar.': 'Industry standard. Dense, isotropic, and easy to machine.',
    'Contrachapado de Abedul:': 'Birch Plywood:', 'Superior en rigidez/peso. Soporta mejor la humedad.': 'Superior stiffness-to-weight ratio and better moisture resistance.',
    'Evitar:': 'Avoid:', 'madera maciza o aglomerado barato — resuena y se deforma.': 'solid wood or inexpensive particleboard—they resonate and deform.',
    'Subwoofers:': 'Subwoofers:', 'mínimo 25mm o doble capa en el baffle frontal.': 'use at least 25mm or a double-layer front baffle.',
    'Proporciones Internas': 'Internal Proportions',
    'Una caja cúbica acumula modos estacionarios en las mismas frecuencias. La calculadora usa la proporción áurea acústica:': 'A cubic enclosure stacks standing modes at the same frequencies. The calculator uses an acoustic golden ratio:',
    'Distribuye los modos de forma suave, evitando picos indeseados en la respuesta interna.': 'This distributes modes more evenly and avoids unwanted peaks in the internal response.',
    'Calcular planos de corte con proporciones áureas': 'Calculate cut plans using golden ratios',
    'La calculadora aplica H:W:D automáticamente a tu Vb': 'The calculator applies H:W:D automatically to your Vb',

    '9. Refuerzos y Amortiguamiento Interno': '9. Internal Bracing and Damping', 'Window Bracing': 'Window Bracing',
    'Las "costillas" internas dividen los paneles grandes en sub-paneles más pequeños, empujando su frecuencia de resonancia hacia arriba donde son más fáciles de absorber.': 'Internal ribs divide large panels into smaller sections, moving their resonant frequency upward where it is easier to absorb.',
    'Doblar el grosor de pared a 36mm puede ser': 'Doubling wall thickness to 36mm can be', 'contraproducente': 'counterproductive',
    ': baja la resonancia del panel al rango auditivo sensible.': ': it can lower panel resonance into a sensitive audible range.',
    'Amortiguamiento (Acoustic Damping)': 'Acoustic Damping',
    'Atenuación de ondas traseras:': 'Rear-wave attenuation:', 'Evita que los medios reboten y atraviesen el cono.': 'Prevents midrange energy from reflecting through the cone.',
    'Efecto termodinámico:': 'Thermodynamic effect:',
    'El material fibroso convierte las variaciones rápidas de presión en calor, reduciendo la velocidad del sonido interior.': 'Fibrous material converts rapid pressure variations into heat, reducing the effective speed of sound inside the enclosure.',
    'Resultado:': 'Result:', 'El altavoz "ve" una caja hasta un': 'The driver behaves as if the enclosure were up to',
    '20% más grande': '20% larger', '— ventaja acústica sin coste de espacio.': '—an acoustic benefit without additional external volume.',
    'Materiales recomendados:': 'Recommended materials:',
    'lana de vidrio, lana de oveja, guata acústica de poliéster. No bloquear el puerto en cajas reflex.': 'fiberglass, sheep wool, or acoustic polyester fill. Do not obstruct the port in reflex enclosures.',

    '10. Matemáticas del Puerto Reflex': '10. Bass Reflex Port Mathematics',
    'Para bajar Fb (graves más profundos):': 'To lower Fb for deeper bass:',
    '1. Aumentar Vb, O BIEN': '1. Increase Vb, OR', '2. Alargar el tubo, O BIEN': '2. Lengthen the port, OR',
    '3. Reducir el área del puerto (con límite de velocidad del aire)': '3. Reduce port area (subject to the air-velocity limit)',
    'Fórmula de longitud del tubo': 'Port Length Formula', 'Derivada de la resonancia de Helmholtz:': 'Derived from Helmholtz resonance:',
    'Donde k es la corrección de extremo (end-correction): 0.614 / 0.732 / 0.850 según la geometría del tubo.': 'Here k is the end correction: 0.614 / 0.732 / 0.850 depending on port geometry.',
    'Límite de velocidad (Chuffing)': 'Air-Velocity Limit (Chuffing)',
    'La velocidad del aire en el puerto debe mantenerse por debajo de': 'Port air velocity should remain below',
    '(idealmente < 12 m/s) para evitar turbulencia audible. Achaflanar ambos extremos del tubo reduce la velocidad efectiva.': '(ideally < 12 m/s) to prevent audible turbulence. Flaring both ends reduces effective air velocity.',
    'Calcular la longitud exacta del puerto': 'Calculate the exact port length',
    'Obtén L en cm, velocidad del aire y diagnóstico de turbulencia': 'Get L in cm, port air velocity, and a turbulence diagnostic',

    '11. Alineamientos Clásicos': '11. Classic Alignments',
    'Interactúa con el control para ver cómo el volumen de la caja cambia la forma de la curva.': 'Use the control to see how enclosure volume changes the response curve.',
    'Volumen Vb:': 'Vb Volume:', 'Óptimo (Butterworth B4)': 'Optimum (Butterworth B4)',
    'Simular con mi altavoz real': 'Simulate with my actual driver',
    'La calculadora aplica las tablas de Thiele a tu Qts específico': 'The calculator applies the Thiele tables to your specific Qts',

    '12. Medición y Pruebas': '12. Measurement and Testing', 'Curva de Impedancia': 'Impedance Curve',
    'Herramienta de diagnóstico más poderosa. Mide los terminales del altavoz montado en la caja:': 'The most powerful diagnostic tool. Measure across the terminals of the driver mounted in the enclosure:',
    'Caja sellada:': 'Sealed enclosure:', 'único pico en Fc.': 'a single peak at Fc.', 'Bass-reflex:': 'Bass reflex:',
    'dos picos gemelos. El valle entre ellos indica Fb real. Ajusta la longitud del tubo hasta que el valle quede en la Fb calculada.': 'two peaks. The valley between them indicates the actual Fb. Adjust port length until the valley matches the calculated Fb.',
    'Herramientas:': 'Tools:',
    'DATS v3 (Dayton Audio), REW + tarjeta de audio, LIMP (Arta software).': 'DATS v3 (Dayton Audio), REW with an audio interface, or LIMP (Arta software).',
    'Medición near-field:': 'Near-field measurement:',
    'Coloca el micrófono a <1 cm del cono para medir la respuesta acústica en graves sin los ecos de la habitación. Combina con la medición del puerto para obtener la respuesta total.': 'Place the microphone less than 1 cm from the cone to measure low-frequency response without room reflections. Combine it with a near-field port measurement to obtain the total response.'
  };

  const originals = new Map();

  function textNodes() {
    const root = document.getElementById('view-enc');
    if (!root) return [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue.trim() || node.parentElement?.closest('svg, script, style')) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    return nodes;
  }

  function replaceText(node, translation) {
    const value = node.nodeValue;
    const start = value.match(/^\s*/)[0];
    const end = value.match(/\s*$/)[0];
    node.nodeValue = `${start}${translation}${end}`;
  }

  window.translateEncyclopedia = function (language) {
    textNodes().forEach(node => {
      if (!originals.has(node)) originals.set(node, node.nodeValue);
      const original = originals.get(node);
      if (language === 'en') {
        const key = original.trim().replace(/\s+/g, ' ');
        if (EN[key]) replaceText(node, EN[key]);
      } else {
        node.nodeValue = original;
      }
    });
  };
}());
