// rface
import React, { useEffect, useState } from "react";
import axios from "axios";
import IndexPage from "./pages/index";
//import RecsPage from "./pages/recsPage";

import {
  Routes,
  Route,
  Navigate,
  useLocation,
  BrowserRouter,
} from "react-router-dom";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { Blogers } from "./pages/Bloger/Blogers";
import { Companies } from "./pages/Company/Companies";
import { Search } from "./pages/Search/Search";
import { Login } from "./pages/Auth/Login";
import { Register } from "./pages/Auth/Register";
import { AuthProvider } from "./context/AuthContext";

const App = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          {/* Protected Routes*/}
          {/* <Route element={<ProtectedRoute />}> */}
          <Route path="/" element={<IndexPage />} />
          <Route path="/blogers" element={<Blogers />} />
          <Route path="/companies" element={<Companies />} />
          <Route path="/search" element={<Search />} />
          {/* </Route> */}
          {/* компании блогеры */}

          {/*<Route path="/stats" element={<StatsPage />} />
        <Route path="/profile" element={<ProfilePage />} />
        <Route path="/сhatBot" element={<ChatBot />} />
        <Route path="*" element={<Navigate to="/" replace />} /> */}
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
};

export default App;
