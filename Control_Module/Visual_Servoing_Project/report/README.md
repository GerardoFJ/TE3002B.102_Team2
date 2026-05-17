# Reporte del proyecto Visual Servoing (Overleaf)

`main.tex` es un documento auto-contenido en formato **IEEE Conference**
(`IEEEtran`), en español, listo para subir a Overleaf.

## Subir a Overleaf

1. En Overleaf: **New Project → Upload Project** y arrastra `main.tex`
   (o todo el contenido de esta carpeta si más adelante agregas figuras).
2. Compilador: **pdfLaTeX** (default en Overleaf).
3. `IEEEtran` ya viene preinstalado en Overleaf — no hay que subirlo.

Si lo compilas localmente:

```bash
pdflatex main.tex
pdflatex main.tex   # segunda pasada para referencias cruzadas
```

## Estructura del reporte

- Resumen (abstract) y palabras clave.
- Introducción y aportes del trabajo.
- Sistema y materiales (hardware, software, frames).
- Calibración hand–eye (gripper → ZED #1).
- MPC para el agarre — incluye formulación matemática completa.
- Alineación con la segunda ZED (centrado + aproximación por tamaño de tag).
- Pipeline integral de 7 etapas.
- Resultados, conclusiones y trabajo futuro.
- Bibliografía (manual, sin .bib externo).

## Notas

- Los **videos de demostración** se referencian como adjuntos en el
  directorio `Control_Module/Visual_Servoing_Project/`. Cuando los
  coloques ahí, no hace falta modificar el `.tex` salvo que quieras
  insertar capturas: añade `\includegraphics{frame.png}` dentro de una
  figura.
- El stack base reutilizado es **RoBorregos `home2`**, citado en la
  bibliografía como `roborregos-home2`.
- Si necesitas cambiar el autor, edita el bloque `\author{...}` arriba
  de `\maketitle` en `main.tex`.
