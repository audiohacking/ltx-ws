import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import App from "./App";
import Layout from "./Layout";
import TrainPage from "./TrainPage";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<App />} />
          <Route path="/train" element={<TrainPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
