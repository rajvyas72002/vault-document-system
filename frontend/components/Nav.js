"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Documents" },
  { href: "/chat", label: "Chat" },
  { href: "/insights", label: "Insights" },
  { href: "/metrics", label: "Metrics" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="nav">
      <strong className="brand">Vault</strong>
      <nav>
        {LINKS.map((l) => (
          <Link key={l.href} href={l.href} className={path === l.href ? "active" : ""}>
            {l.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
