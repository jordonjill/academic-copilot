import { Navigate, useRoutes } from "react-router-dom";

import { WorkspacePage } from "../pages/WorkspacePage";

export function AppRoutes() {
  return useRoutes([
    { path: "/", element: <WorkspacePage /> },
    { path: "*", element: <Navigate to="/" replace /> }
  ]);
}
