More than a Hack 2027 --- requisitos, pistas y alertas
Proyecto: E:\\Hackaton 2027\\FootballAnalytics2027
Repositorio oficial:
https://github.com/metrica-sports/more-than-a-hack-2027
Documento de trabajo basado en lo leído hasta ahora y en la
documentación oficial consultada. No sustituye a los documentos
oficiales. Si hay discrepancias, prevalece el repositorio: DATA.md
para datos, EVALUATION.md para evaluación, RULES.md para reglas y
evaluation/possession_metrics.py para la implementación exacta de
las métricas.

1. Objetivo del reto
Generar una etiqueta por cada uno de los 199.319 fotogramas del
vídeo táctico, a 25 fps:
- H: Barcelona en posesión, balón en juego.
- A: Athletic Club en posesión, balón en juego.
- D: balón muerto; no está en juego.
- X: el sistema se abstiene o el fotograma no tiene etiqueta de
  referencia.
La entrega es un CSV frame,label con 199.319 filas. Se evalúan 158.857
fotogramas de juego; los periodos antes del partido, descanso y después
del final no se puntúan.
2. REQUISITOS --- lo que debemos cumplir
Entregables
1. CSV de posesión con columnas frame,label, 199.319 filas y
   etiquetas H/A/D/X.
2. Repositorio público de código con fuente, dependencias fijadas y un
   comando que regenere la entrega a partir de los datos.
3. Documento técnico breve: método, recursos usados y descartados,
   tratamiento de tracking ausente/ruidoso, reglas de posesión,
   limitaciones y reproducibilidad.
Evaluación
- Métrica principal: spellF1, que evalúa intervalos de posesión.
- macroF1 se informa y es el primer desempate.
- La verdad de referencia pública cubre los primeros cinco minutos de
  juego; la verdad completa es privada.
- Ejecutar evaluation/validate_submission.py antes de entregar.
- El evaluador local documentado es
  python evaluation/run_evaluation.py.
Reproducibilidad y acceso
- El repositorio del equipo debe poder regenerar el CSV.
- Los organizadores deben poder inspeccionar el método y la
  documentación debe explicarlo.
- Los datos del partido se liberan a participantes inscritos que
  aceptaron los términos de uso.
3. PISTAS --- lo que puede ayudar a diseñar el método
Datos proporcionados
- Vídeo táctico y vídeo de retransmisión (broadcast).
- Tracking automático (ATD) de jugadores y balón.
- Smart Tagging automático.
- Homografía para relacionar coordenadas normalizadas del campo con
  píxeles.
El vídeo táctico es la referencia de la entrega. El vídeo broadcast
tiene su propia línea temporal, cortes y repeticiones; no asumir que
está sincronizado con el táctico.
Limitaciones del tracking
- El tracking es automático, sin corrección ni revisión manual.
- Hay 46 espacios para 22 jugadores. Track 1...Track 46 no son
  identidades reales ni números de camiseta.
- Los espacios pueden desaparecer, reaparecer o cambiar de identidad.
  El análisis debe ser colectivo, no depender de identidades
  individuales persistentes.
- La media es de unos 20,5 objetos en los fotogramas evaluados; a
  veces hay más de 22 por duplicaciones.
- Las observaciones ausentes se representan como NaN, tanto para
  jugadores como para el balón.
- El balón aparece en el 56,9 % de todos los fotogramas
  (113.488/199.319) y en el 59,3 % de los evaluados (94.214/158.857).
  Falta en alrededor del 43,1 % del total y el 40,7 % de los
  evaluados.
- Las ausencias no son necesariamente aleatorias: oclusiones,
  aglomeraciones y balones aéreos pueden ocultar el balón en
  situaciones relevantes.
Coordenadas y tiempo
- Una fila de tracking por fotograma; 25 filas equivalen a un segundo.
- x, y están normalizadas entre 0 y 1, no en metros.
- kloppy puede cargar el XML de metadatos junto con el TXT de
  tracking.
- Los metadatos indican 105 × 68 m; games.py usa 105,3 × 68,0 m.
  Verificar las convenciones antes de calcular distancias físicas.
- Alineación documentada: frame_id del ATD va de 1 a 199.319;
  fotograma de vídeo = frame_id - 1. La homografía usa índices
  0--199.318. Smart Tagging usa segundos y se convierte con
  round(seconds * 25).
Reglas conceptuales de posesión
- Proximidad al balón no demuestra por sí sola posesión.
- Un toque o rebote no implica necesariamente un cambio de posesión.
- Durante una disputa por un balón suelto, la posesión puede
  mantenerse atribuida al equipo anterior hasta que el rival obtenga
  el control.
- D significa balón fuera de juego, no simplemente balón quieto.
- Smart Tagging es automático y puede tener falsos positivos y
  negativos: es una señal potencial, no una verdad de referencia.
Baselines
- El repositorio incluye cuatro baselines basados en el jugador
  rastreado más cercano al balón y un voto mayoritario temporal.
- La documentación menciona un baseline con spellF1 = 0,2433, pero
  hay cifras diferentes en algunas tablas/documentos. Ejecutar el
  evaluador local y consultar EVALUATION.md y
  evaluation/possession_metrics.py antes de tomar una cifra como
  definitiva.
- Los cinco minutos públicos sirven para comprobar el flujo de datos y
  evaluación; no representan el ranking final. Las reglas indican que
  no se deben usar para entrenar o ajustar el método.
4. ALERTAS --- errores que debemos evitar
- No asumir que el balón tiene posición en cada fotograma. Diseñar
  para los NaN desde el principio.
- No tratar los números de Track como identidades de futbolistas.
- No confundir balón no observado con balón muerto. NaN no
  equivale a D.
- No confundir proximidad con control.
- No interpolar a ciegas todos los huecos. Pases, rebotes y
  contactos pueden invalidar una trayectoria lineal.
- No mezclar las líneas temporales del táctico y el broadcast.
- No entregar 164.488 filas: ese es el número de registros ATD del
  broadcast; la entrega debe cubrir los 199.319 fotogramas tácticos.
- No asumir que MIT cubre los vídeos/datos. El código y
  documentación tienen licencia MIT, pero los datos del partido están
  sujetos a los términos específicos de acceso y uso.
- No anotar manualmente las etiquetas del partido de la
  competición. Las etiquetas entregadas deben generarse
  algorítmicamente.
- No usar la verdad privada ni material derivado de ella, ni datos
  humanos del partido como eventos, tracking corregido o anotaciones.
- No subir vídeos, datasets, información confidencial ni extractos
  restringidos a LLM alojados en la nube de terceros. Mantener el
  material en el entorno local y respetar los términos de uso.
- No modificar los datos originales. Auditar primero y guardar
  transformaciones por separado.
- No optimizar solo macroF1 o la exactitud por fotograma. La
  métrica principal es spellF1.
5. Lista de lectura oficial
Marca cada documento solo cuando se haya leído completo y anotado sus
requisitos.
- [x] README.md --- visión general, estructura, entorno y orden de
  lectura.
- [x] CITATION.cff --- metadatos para citar el benchmark; no es
  código ni una lista de requisitos de entrega.
- [ ] CHALLENGE.md (quizá lo tienes como Desafio.md) ---
  definición de posesión, etiquetas y alcance; comprobar el nombre
  exacto.
- [ ] DATA.md --- archivos, formatos, coordenadas, alineación y
  limitaciones.
- [ ] EVALUATION.md --- métricas y evaluación.
- [ ] SUBMISSION.md --- formato y proceso de entrega.
- [ ] RULES.md (quizá lo tienes como reglas.md) --- permisos y
  prohibiciones; comprobar el nombre exacto.
- [ ] FAQ.md --- preguntas frecuentes.
- [ ] OVERVIEW.md (quizá lo tienes como Leame.md) --- contexto,
  calendario y formato del reto.
- [ ] evaluation/possession_metrics.py --- implementación exacta de
  las métricas.
- [ ] evaluation/validate_submission.py --- validación de la
  entrega.
- [ ] games.py --- rutas, dimensiones y alineación.
- [ ] starter/01_smart_tagging.ipynb --- inspección de Smart
  Tagging.
- [ ] starter/02_tracking_atd.ipynb --- carga y visualización del
  tracking.
- [ ] starter/03_first_submission.ipynb --- datos a primera entrega
  evaluable.
- [ ] starter/baseline.py --- algoritmo de referencia.
- [ ] requirements.txt y requirements-starter.txt --- dependencias
  del evaluador y los notebooks.
6. Plan antes de programar el algoritmo
1. Terminar la lectura de los documentos oficiales y actualizar esta
   lista.
2. Resolver discrepancias consultando el archivo oficial
   correspondiente y el código del evaluador.
3. Inspeccionar los loaders y notebooks sin modificar los datos
   originales.
4. Auditar localmente dimensiones, valores ausentes, continuidad,
   velocidades, alineación y pertenencia al equipo.
5. Ejecutar el baseline y el evaluador para obtener una referencia
   reproducible.
6. Diseñar hipótesis y características colectivas; probar una
   modificación cada vez y registrar resultados.
7. Conservar solo cambios reproducibles que mejoren spellF1 y
   respeten las reglas.
8. Validar la salida y comprobar que el repositorio regenera
   exactamente el CSV entregado.
7. Decisiones pendientes
- [ ] Confirmar nombres y contenido completo de los documentos
  oficiales.
- [ ] Verificar definición exacta de spellF1, tratamiento de X y
  coincidencia de intervalos.
- [ ] Confirmar cómo se representa la pertenencia al equipo y las
  detecciones ausentes.
- [ ] Confirmar unidades y convenciones de coordenadas antes de
  calcular distancias.
- [ ] Verificar las señales de Smart Tagging y su alineación temporal.
- [ ] Diseñar una auditoría de datos que no altere los archivos
  originales.
- [ ] Ejecutar y registrar las métricas locales de los baselines.
8. Enlaces oficiales
- Repositorio
- README
- CHALLENGE
- DATA
- EVALUATION
- SUBMISSION
- RULES
- FAQ
- OVERVIEW