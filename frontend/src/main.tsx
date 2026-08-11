import {
  StrictMode,
} from "react";
import {
  createRoot,
} from "react-dom/client";

import "ol/ol.css";

import App from "./App";
import "./styles.css";

const rootElement =
  document.getElementById(
    "root",
  );

if (
  rootElement ===
  null
) {
  throw new Error(
    "VoltSight root element was not found.",
  );
}

createRoot(
  rootElement,
).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
