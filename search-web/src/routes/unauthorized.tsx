import Unauthorized from "@/components/Unauthorized";
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/unauthorized')({
  component: Unauthorize
})

function Unauthorize() {
  return <Unauthorized />;
}
