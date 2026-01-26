import { PageLoader } from "@/components/PageLoader";
import { createFileRoute, Navigate, Outlet } from "@tanstack/react-router";
import { useEffect } from "react";
import { useAuth } from "react-oidc-context";
import { useRoles } from "@/contexts/AuthContext";

export const Route = createFileRoute("/_authenticated")({
  component: Auth,
});

function Auth() {
  const { isAuthenticated, signinRedirect, isLoading: isUserAuthLoading } = useAuth();
  const { hasAnyRole, isLoading: isRolesLoading } = useRoles();

  const isLoading = isUserAuthLoading || isRolesLoading;

  useEffect(() => {
    if (!isAuthenticated && !isUserAuthLoading) {
      const returnPath = window.location.pathname + window.location.search;
      localStorage.setItem("authReturnPath", returnPath);
      signinRedirect();
    }
  }, [isAuthenticated, isUserAuthLoading, signinRedirect]);

  if (isLoading) {
    return <PageLoader />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/" />;
  }

  if (!hasAnyRole) {
    return <Navigate to="/unauthorized" />;
  }

  return <Outlet />;
}
