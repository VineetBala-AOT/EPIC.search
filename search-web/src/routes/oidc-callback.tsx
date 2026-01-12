import { createFileRoute, Navigate } from "@tanstack/react-router";
import { useAuth } from "react-oidc-context";
import { PageLoader } from "@/components/PageLoader";
import { useRoles } from "@/contexts/AuthContext";
import { useEffect, useState } from "react";

export const Route = createFileRoute("/oidc-callback")({
  component: OidcCallback,
});

function OidcCallback() {
  const { error: getAuthError, user: kcUser, isLoading } = useAuth();
  const { hasAnyRole } = useRoles();
  const [navTarget, setNavTarget] = useState<string | null>(null);

  useEffect(() => {
    if (kcUser && hasAnyRole !== undefined) {
      const saved = localStorage.getItem("authReturnPath") ?? "/search";
      setNavTarget(saved);
    }
  }, [kcUser, hasAnyRole]);

  if (getAuthError) {
    return <Navigate to="/error" />;
  }

  if (isLoading || navTarget === null) {
    return <PageLoader />;
  }

  return <Navigate to={navTarget} />;
}
