import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pneumonia Detection AI",
  description: "Chest X-ray pneumonia detection with Next.js, Flask, and TensorFlow"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
